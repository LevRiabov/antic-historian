import { PageStub } from "@/components/PageStub";

export function Chat() {
  return (
    <PageStub title="Ask" mockup="chat-05-hybrid.html">
      Streaming cited answers (fast / deep mode) over POST /ask. The typed SSE
      client is ready in src/lib/sse.ts.
    </PageStub>
  );
}
