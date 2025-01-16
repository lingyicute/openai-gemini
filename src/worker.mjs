import { Buffer } from "node:buffer";

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      return new Response(
        JSON.stringify({ error: err.message }),
        fixCors({ status: err.status ?? 500 })
      );
    };
    try {
      const auth = request.headers.get("Authorization");
      const apiKey = auth?.split(" ")[1];
      const assert = (success) => {
        if (!success) {
          throw new HttpError(
            "The specified HTTP method is not allowed for the requested resource",
            400
          );
        }
      };
      const { pathname } = new URL(request.url);
      switch (true) {
        case pathname.endsWith("/chat/completions"):
          assert(request.method === "POST");
          return handleCompletions(await request.json(), apiKey).catch(
            errHandler
          );
        case pathname.endsWith("/embeddings"):
          assert(request.method === "POST");
          return handleEmbeddings(await request.json(), apiKey).catch(
            errHandler
          );
        case pathname.endsWith("/models"):
          assert(request.method === "GET");
          return handleModels(apiKey).catch(errHandler);
        default:
          throw new HttpError("404 Not Found", 404);
      }
    } catch (err) {
      return errHandler(err);
    }
  },
};

class HttpError extends Error {
  constructor(message, status) {
    super(message);
    this.name = this.constructor.name;
    this.status = status;
  }
}

const fixCors = ({ headers = {}, status, statusText }) => {
  const fixedHeaders = new Headers(headers);
  fixedHeaders.set("Access-Control-Allow-Origin", "*");
  fixedHeaders.set("Access-Control-Allow-Methods", "*");
  fixedHeaders.set("Access-Control-Allow-Headers", "*");
  return { headers: fixedHeaders, status, statusText };
};

const handleOPTIONS = async () => {
  return new Response(null, {
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "*",
      "Access-Control-Allow-Headers": "*",
    },
  });
};

const BASE_URL = "https://generativelanguage.googleapis.com";
const API_VERSION = "v1beta";

const API_CLIENT = "genai-js/0.21.0";
const makeHeaders = (apiKey, more) => ({
  "x-goog-api-client": API_CLIENT,
  ...(apiKey && { "x-goog-api-key": apiKey }),
  ...more,
});

async function handleModels(apiKey) {
  const response = await fetch(`${BASE_URL}/${API_VERSION}/models`, {
    headers: makeHeaders(apiKey),
  });
  let { body } = response;
  if (response.ok) {
    const { models } = JSON.parse(await response.text());
    body = JSON.stringify(
      {
        object: "list",
        data: models.map(({ name }) => ({
          id: name.replace("models/", ""),
          object: "model",
          created: 0,
          owned_by: "",
        })),
      },
      null,
      "  "
    );
  }
  return new Response(body, fixCors({ headers: response.headers }));
}

const DEFAULT_EMBEDDINGS_MODEL = "text-embedding-004";
async function handleEmbeddings(req, apiKey) {
  if (typeof req.model !== "string") {
    throw new HttpError("model is not specified", 400);
  }
  if (!Array.isArray(req.input)) {
    req.input = [req.input];
  }
  let model;
  if (req.model.startsWith("models/")) {
    model = req.model;
  } else {
    req.model = DEFAULT_EMBEDDINGS_MODEL;
    model = "models/" + req.model;
  }
  const response = await fetch(
    `${BASE_URL}/${API_VERSION}/${model}:batchEmbedContents`,
    {
      method: "POST",
      headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
      body: JSON.stringify({
        requests: req.input.map((text) => ({
          model,
          content: { parts: { text } },
          outputDimensionality: req.dimensions,
        })),
      }),
    }
  );
  let { body } = response;
  if (response.ok) {
    const { embeddings } = JSON.parse(await response.text());
    body = JSON.stringify(
      {
        object: "list",
        data: embeddings.map(({ values }, index) => ({
          object: "embedding",
          index,
          embedding: values,
        })),
        model: req.model,
      },
      null,
      "  "
    );
  }
  return new Response(body, fixCors({ headers: response.headers }));
}

const DEFAULT_MODEL = "gemini-1.5-pro-latest";
async function handleCompletions(req, apiKey) {
  let model = DEFAULT_MODEL;
  switch (true) {
    case typeof req.model !== "string":
      break;
    case req.model.startsWith("models/"):
      model = req.model.substring(7);
      break;
    case req.model.startsWith("gemini-"):
    case req.model.startsWith("learnlm-"):
      model = req.model;
  }
  const TASK = req.stream ? "streamGenerateContent" : "generateContent";
  const id = generateChatcmplId();
  const responseJson = processCompletionsResponse({}, model, id, req);
  return new Response(responseJson, fixCors({}));
}

const processCompletionsResponse = (data, model, id, req) => {
  return JSON.stringify(
    {
      id,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: JSON.stringify(req, null, 2), // Ensure `req` is JSON stringified
          },
          finish_reason: "stop",
        },
      ],
      usage: null, // No usage data is available in this mock response
    },
    null,
    2
  );
};

const generateChatcmplId = () => {
  return "chatcmpl-" + Math.random().toString(36).substring(2, 15);
};
