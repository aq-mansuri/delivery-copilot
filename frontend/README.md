# Frontend

React + Vite. Builds straight into `../app/api/static/`, which FastAPI serves —
so `npm run build` followed by `uvicorn` is the whole deploy, with no copy step
to forget.

    npm install
    npm run build          # production bundle into app/api/static/
    npm run dev            # dev server on 5173, proxying the API on 8000

`npm run dev` proxies `/ask`, `/proposals` and `/health` to `localhost:8000`, so
the browser stays on one origin and there is no CORS configuration that exists
only for development.

## Structure

    src/api.js          server calls and SSE framing
    src/components.jsx  presentational pieces
    src/App.jsx         state and orchestration
    src/styles.css      tokens and layout

SSE is parsed by hand in `api.js` because `EventSource` only supports GET and
`/ask` is a POST. That belongs in the API layer, not in a component.

Answer text is split on the citation pattern and rendered as React children
rather than injected as HTML. The text comes from a model, and
`dangerouslySetInnerHTML` on model output is how a prompt injection becomes a
script tag.
