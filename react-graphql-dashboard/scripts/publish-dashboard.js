#!/usr/bin/env node
/**
 * Publish script for react-graphql-dashboard.
 *
 * Usage:
 *   npm run publish              # bump patch version (default) and publish
 *   npm run publish:patch        # same as above
 *   npm run publish:minor        # bump minor version and publish
 *   npm run publish:major        # bump major version and publish
 *
 * The script performs the following steps:
 *  1. Runs full build for all workspaces (frontend & backend).
 *  2. Bumps the package.json version (patch|minor|major).
 *  3. Publishes the root package to npm with public access.
 *  4. Pushes git commit & tag so that CI and changelogs stay in sync.
 *
 * You can set the environment variable `NPM_REGISTRY` to publish to a custom
 * registry (defaults to the public npm registry).
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const rootDir = path.resolve(__dirname, '..');

function exec(cmd) {
  console.log(`\n$ ${cmd}`);
  execSync(cmd, { stdio: 'inherit', cwd: rootDir });
}

function main() {
  const bumpType = process.argv[2] || 'patch';
  const allowed = ['patch', 'minor', 'major'];
  if (!allowed.includes(bumpType)) {
    console.error(`Invalid version bump type \"${bumpType}\". Use one of ${allowed.join(', ')}.`);
    process.exit(1);
  }

  // Ensure working directory is clean before publishing
  const changes = execSync('git status --porcelain', { cwd: rootDir }).toString().trim();
  if (changes) {
    console.error('⚠️  Your working tree has uncommitted changes. Commit or stash them before publishing.');
    process.exit(1);
  }

  // 1. Build the project (frontend + backend)
  exec('npm run build');

  // 2. Bump version (this also creates a git tag)
  exec(`npm version ${bumpType} -m "chore(release): %s"`);

  // 3. Publish to npm
  const registry = process.env.NPM_REGISTRY ? `--registry ${process.env.NPM_REGISTRY}` : '';
  exec(`npm publish --access public ${registry}`);

  // 4. Push commit and tags to origin
  exec('git push --follow-tags');

  // 5. Success message
  const pkg = JSON.parse(fs.readFileSync(path.join(rootDir, 'package.json'), 'utf-8'));
  console.log(`\n✅ Successfully published react-graphql-dashboard v${pkg.version}`);
}

main();
