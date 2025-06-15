http://localhost:4000/ (Apollo Server / GraphQL Backend):
This is your data hub and brain.
It hosts your GraphQL API.
Its job is to receive queries from your frontend, process them (e.g., fetch data from a database, perform calculations, interact with other services), and send back the requested data.
The Apollo Sandbox UI you're seeing in the screenshot is a development tool built into Apollo Server that allows you to explore your GraphQL schema and test queries directly against your backend. This is very useful for backend development and debugging.
All the "consciousness" related data, performance metrics, user data, etc., are served from here.

http://localhost:3001/ (or http://localhost:3002/ - Vite Development Server / React Frontend):
This is your user interface (UI).
It serves your React application to the browser. This is what the end-user actually sees and interacts with.
Your React components, including the Dashboard.tsx and the ConsciousnessVisualization.tsx, are bundled and served by Vite from this address.
This frontend application makes GraphQL queries to your backend server (http://localhost:4000/) to get the data it needs to display. For example, when the dashboard loads, it will query the backend for metrics, and when the consciousness visualization runs, it will query for consciousness-related data points.
Vite also provides features like Hot Module Replacement (HMR), so when you make changes to your frontend code, the browser updates automatically without a full page reload, speeding up development.

# Dependency Map – “resume-2025-novabright” repo (June 2025)

Below is a consolidated view of every explicit dependency definition I could find, grouped by language / package-manager and by workspace.

---

## 1. Python stack

Source file     | Manager | Explicit runtime deps  
-----------------|---------|---------------------------------------------  
`pixi.toml`     | pixi / conda-mamba |  
• openai >= 1.84,<2  
• pip >= 25.1,<26  
• max >= 25.4.0.dev2025060721,<26   *(Modular Max nightly)*  
• numpy >= 2.2.6,<3  
• matplotlib >= 3.10.3,<4  

Notes  
• Pixi is a **cross-platform Conda environment manager** (similar to Mamba).  
• No `requirements.txt`, `Pipfile`, or `pyproject.toml` files were found elsewhere, suggesting this is the single authoritative environment.  
• `start-local-server.py` is the only first-party Python source at repo root; the many Python files under `.magic/` are test fixtures that ship with the Conda environment and can be ignored for dependency purposes.

---

## 2. Node / TypeScript monorepo  
Path (workspace) | package manager | Runtime dependencies | Dev-only dependencies  
-----------------|-----------------|---------------------|-----------------------  
`react-graphql-dashboard/` (root) | npm workspaces | lodash, uuid | concurrently, eslint (+plugins), prettier, typescript, @types/node, @typescript-eslint/*  
`react-graphql-dashboard/frontend` | npm | • @apollo/client (+ streaming) • @reduxjs/toolkit • @tanstack/react-query • graphql • crypto-hash • framer-motion • plotly.js / react-plotly • react 18 • react-router-dom • react-hook-form • recharts • three | vitest (+ ui, coverage), vite (+ react plugin & bundle-analyzer), ESLint stack, Tailwind build chain, TS tooling, Testing Library types, etc.  
`react-graphql-dashboard/backend` | npm | • @apollo/server (+ cache plugins) • aws-sdk v3 clients (DynamoDB, ElastiCache) • graphql-redis-subscriptions • ioredis • dataloader • graphql • compression, helmet, cors • uuid, lodash • winston | jest, ts-jest, nodemon, serverless (& plugins), ESLint, TypeScript, @types/*  
`react-graphql-dashboard/infrastructure` | npm | AWS CDK v2 libs (`aws-cdk-lib`, constructs) + experimental ApigwV2 alpha modules | CDK CLI, jest, ESLint, TypeScript, @types/*  

Shared characteristics  
• Target Node ≥ 18 (declared in all `engines`).  
• Workspace root offers convenience scripts: `npm run dev`, `build`, `deploy`, etc., delegating to sub-workspaces.  
• No lockfiles committed in tree; consider adding `package-lock.json` / `pnpm-lock.yaml` / `yarn.lock` for reproducibility.

---

## 3. Other ecosystems

None detected (no Cargo.toml, go.mod, Gemfile, etc.).

---

## 4. Visual summary

```
📦 resume-2025-novabright
├─ pixi.toml  (Conda env → Python deps)
└─ react-graphql-dashboard/
   ├─ package.json           (root workspace deps)
   ├─ frontend/
   │   └─ package.json       (React UI deps)
   ├─ backend/
   │   └─ package.json       (GraphQL API deps)
   └─ infrastructure/
       └─ package.json       (AWS CDK deps)
```

---

## 5. Suggested next steps

1. Lockfiles  
   • Commit `pixi.lock` / `conda-lock.yml` and `package-lock.json` (or `pnpm-lock.yaml`) for deterministic builds.  
2. Dependency health  
   • Run `npm outdated --workspaces` and `pixi run conda check --channel conda-forge` (or similar) to spot stale versions.  
3. Security scanning  
   • Integrate `npm audit --workspaces`, `osv-scanner` (for Python via requirements export), or GitHub Dependabot.  
4. CI matrix  
   • Add CI jobs that install the pixi environment and run `npm run type-check && npm test` in each workspace.  

Let me know if you’d like a deeper dive into any particular dependency group, transitive-license inventory, or automated SBOM generation.