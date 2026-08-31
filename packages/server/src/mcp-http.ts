import { randomUUID } from 'node:crypto';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { displayNameFromClientName } from '@inkeep/open-knowledge-core';
import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import { validateAgentId } from './agent-id.ts';
import type { Config } from './config/schema.ts';
import { MCP_SERVER_NAME } from './constants.ts';
import type { LocalApiDispatch } from './http/local-api-dispatch.ts';
import {
  type AgentIdentity,
  MCP_CONNECTION_ID_HEADER,
  MCP_HOSTED_AGENT_HEADER,
  sanitizeClientName,
} from './mcp/agent-identity.ts';
import { installJsonSchemaDialect } from './mcp/json-schema-dialect.ts';
import { installPrettyZodErrors } from './mcp/pretty-zod-errors.ts';
import { registerAllTools } from './mcp/tools/index.ts';
import { resolveWithinRoot } from './mcp/tools/path-safety.ts';
import { RUNTIME_VERSION } from './version-constants.ts';

interface McpHttpSession {
  transport: StreamableHTTPServerTransport;
  server: McpServer;
}

export interface McpHttpHandlerOptions {
  contentDir: string;
  projectDir?: string;
  /**
   * The project's loaded `Config`. Tool handlers read settings off this object
   * (e.g. `config.content.dir`). The fields used downstream MUST match what
   * the user wrote in `.ok/config.yml` — never fabricate a synthetic config
   * here.
   */
  config: Config;
  /** Returns the base URL of this running HTTP server, without the `/mcp` suffix. */
  getServerUrl: () => string;
  /**
   * In-process `/api/*` dispatch (`ServerInstance.localApi`). When present,
   * tools whose endpoints are backed by the extracted capability services
   * invoke the handler in-process instead of round-tripping HTTP to their
   * own listener; paths outside the collapsed allowlist fall back to HTTP.
   * The stdio `ok mcp` proxy never has this — it talks to a separate server
   * process and stays on HTTP.
   */
  localApi?: LocalApiDispatch;
  log?: {
    info?: (obj: object, msg: string) => void;
    warn?: (obj: object, msg: string) => void;
    error?: (obj: object, msg: string) => void;
  };
  /** Deprecated: HTTP MCP is stateless-only in the EDU fork. */
  sessionTtlMs?: number;
  /** Deprecated: HTTP MCP is stateless-only in the EDU fork. */
  maxSessions?: number;
  /** Deprecated: HTTP MCP is stateless-only in the EDU fork. */
  stateless?: boolean;
}

export interface McpHttpHandler {
  handle: (req: IncomingMessage, res: ServerResponse) => Promise<void>;
  close: () => Promise<void>;
}

function firstHeader(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

function writePlain(res: ServerResponse, statusCode: number, message: string): void {
  if (res.writableEnded) return;
  res.statusCode = statusCode;
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.end(message);
}

function createSessionServer(
  opts: McpHttpHandlerOptions,
  transport: StreamableHTTPServerTransport,
  forwardedConnectionId: string | undefined,
  isHostedAgent: boolean,
): McpHttpSession {
  const config = opts.config;
  // No `instructions` handshake — see startGlobalMcpServer. The
  // project skill is the single steering channel; the HTTP server only ever
  // runs inside an OK project, where that skill is installed.
  const server = new McpServer({
    name: MCP_SERVER_NAME,
    version: RUNTIME_VERSION,
  });
  installPrettyZodErrors(server);

  // `connectionId` is the only stable per-session disambiguator when multiple
  // clients report the same MCP `clientInfo.name`, such as two Claude
  // instances connected to the same `ok start` server.
  //
  // When the `ok mcp` shim forwards its keepalive WS connectionId via the
  // `MCP_CONNECTION_ID_HEADER`, adopt it so the keepalive's 3 s
  // `bumpPresenceTs` heartbeat and on-close `clearPresence` (both keyed by
  // the keepalive id) operate on the same broadcaster entry that write
  // handlers create with `agentId: identity.connectionId`. Without this
  // unification the icon flickers per tool call (5 s TTL) instead of
  // staying visible for the lifetime of the MCP session.
  const connectionId = forwardedConnectionId ?? randomUUID();
  const identityRef: { current: AgentIdentity } = {
    current: {
      connectionId,
      displayName: connectionId,
      colorSeed: connectionId,
    },
  };

  server.server.oninitialized = () => {
    const clientInfo = server.server.getClientVersion();
    const name = sanitizeClientName(clientInfo?.name, connectionId);
    identityRef.current = {
      connectionId,
      clientInfo: clientInfo ? { name, version: clientInfo.version } : undefined,
      displayName: displayNameFromClientName(name),
      colorSeed: name,
    };
  };

  // The configured project root is the trust boundary for every tool call:
  // an explicit `cwd` arg from the MCP client must lexically resolve to,
  // or under, this directory. Without this gate the tools accept any
  // absolute path the caller hands them (`cwd: "/etc"`) and exec /
  // search reach outside the configured content scope —
  // see resolveWithinRoot() for the containment contract. The MCP-roots
  // negotiation (one-advertised-root → default) lives in the stdio
  // proxy; the HTTP MCP server here always anchors on the configured
  // projectDir / contentDir.
  const configuredRoot = opts.projectDir ?? opts.contentDir;
  registerAllTools(server, {
    serverUrl: async () => opts.getServerUrl(),
    localApi: opts.localApi,
    resolveCwd: async (explicit?: string) => {
      if (explicit === undefined) return configuredRoot;
      const result = resolveWithinRoot(configuredRoot, explicit);
      if (!result.ok) {
        throw new Error(
          `cwd "${explicit}" is not within the configured project root: ${result.reason}`,
        );
      }
      return result.abs;
    },
    config,
    identityRef,
    isHostedAgent,
  });

  // After `registerAllTools`, not before: the SDK installs its `tools/list`
  // handler lazily on the first `registerTool`.
  installJsonSchemaDialect(server);

  return { server, transport };
}

/**
 * Create a stateless Streamable HTTP MCP endpoint handler for `POST /mcp`.
 *
 * The MCP implementation lives in the running project server. A stdio `ok mcp`
 * process should only proxy JSON-RPC frames to this endpoint; it should not
 * register tools itself.
 */
export function createMcpHttpHandler(opts: McpHttpHandlerOptions): McpHttpHandler {
  async function handleStateless(req: IncomingMessage, res: ServerResponse): Promise<void> {
    if (req.method !== 'POST') {
      writePlain(res, 400, 'Missing MCP session. Initialize with POST /mcp first.');
      return;
    }

    const rawConnectionIdHeader = firstHeader(req.headers[MCP_CONNECTION_ID_HEADER]);
    const forwardedConnectionId = validateAgentId(rawConnectionIdHeader) ?? undefined;
    if (rawConnectionIdHeader !== undefined && forwardedConnectionId === undefined) {
      opts.log?.warn?.(
        { headerLength: rawConnectionIdHeader.length },
        'MCP HTTP forwarded connectionId header failed validation; falling back to randomUUID',
      );
    }

    const isHostedAgent = firstHeader(req.headers[MCP_HOSTED_AGENT_HEADER]) === '1';
    const transport = new StreamableHTTPServerTransport({
      sessionIdGenerator: undefined,
      enableJsonResponse: true,
    });
    const session = createSessionServer(opts, transport, forwardedConnectionId, isHostedAgent);
    try {
      await session.server.connect(transport);
      await transport.handleRequest(req, res);
    } finally {
      await Promise.allSettled([session.server.close(), transport.close()]);
    }
  }

  return {
    async handle(req, res): Promise<void> {
      // EDU fork: always stateless. Stateful Streamable HTTP sessions are a
      // footgun for cloud clients that cache dead MCP-Session-Id values.
      await handleStateless(req, res);
    },

    async close(): Promise<void> {
      // Stateless requests create per-request transports and close them in
      // handleStateless(), so there is no handler-level session pool to drain.
    },
  };
}
