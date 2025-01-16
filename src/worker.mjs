import { Buffer } from "node:buffer";

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") {
      return handleOPTIONS();
    }
    const errHandler = (err) => {
      console.error(err);
      return new Response(err.message, fixCors({ status: err.status ?? 500 }));
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

const fixCors = ({ headers, status, statusText }) => {
  headers = new Headers(headers);
  headers.set("Access-Control-Allow-Origin", "*");
  return { headers, status, statusText };
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
  return new Response(body, fixCors(response));
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
  return new Response(body, fixCors(response));
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
  let url = `${BASE_URL}/${API_VERSION}/models/${model}:${TASK}`;
  if (req.stream) {
    url += "?alt=sse";
  }
  const response = await fetch(url, {
    method: "POST",
    headers: makeHeaders(apiKey, { "Content-Type": "application/json" }),
    body: JSON.stringify(await transformRequest(req)),
  });

  let body = response.body;
  if (response.ok) {
    let id = generateChatcmplId();
    if (req.stream) {
      body = response.body
        .pipeThrough(new TextDecoderStream())
        .pipeThrough(
          new TransformStream({
            transform: parseStream,
            flush: parseStreamFlush,
            buffer: "",
          })
        )
        .pipeThrough(
          new TransformStream({
            transform: toOpenAiStream,
            flush: toOpenAiStreamFlush,
            streamIncludeUsage: req.stream_options?.include_usage,
            model,
            id,
            last: [],
          })
        )
        .pipeThrough(new TextEncoderStream());
    } else {
      body = await response.text();
      body = processCompletionsResponse(JSON.parse(body), model, id, req);
    }
  }
  return new Response(body, fixCors(response));
}

const transformMessages = async (messages) => {
  return {
    messages: messages.map((message) => ({
      role: message.role || "user",
      content: message.content || "",
    })),
  };
};

const transformRequest = async (req) => ({
  ...(await transformMessages(req.messages)),
  generationConfig: transformConfig(req),
});

const processCompletionsResponse = (data, model, id, req) => {
  const responseJson = {
    id,
    choices: data.candidates.map((cand) => ({
      index: cand.index || 0,
      message: {
        role: "assistant",
        content: JSON.stringify(req), // 包含入站 JSON 作为“模型的回答”
      },
      logprobs: null,
      finish_reason: cand.finishReason,
    })),
    created: Math.floor(Date.now() / 1000),
    model,
    object: "chat.completion",
    usage: transformUsage(data.usageMetadata),
  };
  return JSON.stringify(responseJson, null, "  ");
};
