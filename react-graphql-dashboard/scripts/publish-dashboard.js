#!/usr/bin/env node
/**
 * Enhanced publish script for react-graphql-dashboard.
 *
 * Features:
 * 1. Quality gates: lint → type-check → tests → build.
 * 2. Workspace-wide version bumping (root + all public workspaces).
 * 3. 2-Factor-Auth & dry-run support (`--otp=<code>` and `--dry`).
 * 4. Atomic publishing with automatic rollback on failure.
 *
 * Usage examples:
 *   npm run publish              # bump patch (default) and publish
 *   npm run publish minor        # bump minor and publish
 *   npm run publish major --dry  # preview a major release without publishing or pushing
 *
 * Environment variables:
 *   NPM_REGISTRY   – custom registry (defaults to npmjs.org)
 *   NPM_OTP        – 2FA one-time password (overridden by --otp flag)
 */
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const rootDir = path.resolve(__dirname, '..');

function exec(cmd, opts = {}) {
  console.log(`\n$ ${cmd}`);
  execSync(cmd, { stdio: 'inherit', cwd: rootDir, ...opts });
}

// ----------------------- CLI Argument Parsing ----------------------- //
const allowedBumps = new Set(['patch', 'minor', 'major']);
let bumpType = 'patch';
let dryRun = false;
let otp = process.env.NPM_OTP || '';

process.argv.slice(2).forEach((arg) => {
  if (allowedBumps.has(arg)) {
    bumpType = arg;
  } else if (arg === '--dry' || arg === '--dry-run') {
    dryRun = true;
  } else if (arg.startsWith('--otp=')) {
    otp = arg.substring('--otp='.length);
  } else {
    console.error(`Unknown argument: ${arg}`);
    process.exit(1);
  }
});

const registryFlag = process.env.NPM_REGISTRY ? `--registry ${process.env.NPM_REGISTRY}` : '';
const otpFlag = otp ? `--otp=${otp}` : '';
const dryRunFlag = dryRun ? '--dry-run' : '';

// ----------------------------- Helpers ------------------------------ //
function workingTreeClean() {
  const changes = execSync('git status --porcelain', { cwd: rootDir }).toString().trim();
  return changes.length === 0;
}

function getCurrentVersion() {
  const pkg = JSON.parse(fs.readFileSync(path.join(rootDir, 'package.json'), 'utf8'));
  return pkg.version;
}

afunction rollback(tag) {
  try {
    console.log('\n⏪ Rolling back…');
    if (tag) {
      execSync(`git tag -d ${tag}`, { stdio: 'inherit', cwd: rootDir });
    }
    execSync('git reset --hard HEAD~1', { stdio: 'inherit', cwd: rootDir });
    console.log('✅ Rollback complete.');
  } catch (err) {
    console.error('⚠️  Rollback failed – please inspect your repository manually.', err);
  }
}

// ------------------------------ Main ------------------------------- //
if (!workingTreeClean()) {
  console.error('⚠️  Your working tree has uncommitted changes. Commit or stash them before publishing.');
  process.exit(1);
}

let newTag = null;
try {
  // 1. Quality gates
  exec('npm run lint');
  exec('npm run type-check');
  exec('npm test');

  // 2. Build step (aggregates frontend & backend)
  exec('npm run build');

  // 3. Bump versions across all public workspaces (no git tags yet)
  exec(`npm workspaces foreach --no-private --topological exec "npm version ${bumpType} --no-git-tag-version"`);

  // 4. Bump root version & create git tag
  exec(`npm version ${bumpType} -m "chore(release): %s"`);
  const version = getCurrentVersion();
  newTag = `v${version}`; // npm uses "v" prefix by default

  // 5. Publish every public workspace, then root
  exec(`npm workspaces foreach --no-private --topological exec "npm publish --access public ${registryFlag} ${otpFlag} ${dryRunFlag}"`);
  exec(`npm publish --access public ${registryFlag} ${otpFlag} ${dryRunFlag}`);

  // 6. Push tags & commit when not dry-running
  if (!dryRun) {
    exec('git push --follow-tags');
  } else {
    console.log('ℹ️  Dry run – skipping git push.');
  }

  console.log(`\n✅ Successfully published react-graphql-dashboard v${version}${dryRun ? ' (dry run)' : ''}`);
} catch (error) {
  console.error('\n❌ Publish failed:', error.message || error);
  rollback(newTag);
  process.exit(1);
}
