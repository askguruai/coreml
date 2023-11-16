# ML Service

Provides embeddings, Q&A and summaries for AskGuru. In it's current form, it is a messy OpenAI API wrapper.

## Deployment

```bash
sudo -E docker compose up -d --build
```

Requires `OPENAI_API_KEY` to be set.
