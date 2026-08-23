// semantic-release configuration.
//
// Triggered by .github/workflows/release.yml on push to main.
// Reads merged-PR labels to compute the next version bump, then creates a
// bare-semver GitHub Release with auto-generated notes from PR titles.
//
// Label map (only labels listed here trigger a release):
//   - breaking → major
//   - feature  → minor
//   - bug      → patch
// Any other label (chore, documentation, etc.) is ignored. A PR with no
// release-triggering label contributes no version bump. If a release window
// contains only non-mapped labels, semantic-release exits with
// "no release published" — this is expected, not an error.
//
// When a PR carries multiple release-triggering labels (e.g., breaking +
// feature), the highest bump wins: major > minor > patch.

module.exports = {
  branches: ['main'],
  tagFormat: '${version}',
  plugins: [
    [
      '@bobvanderlinden/semantic-release-pull-request-analyzer',
      {
        labels: {
          breaking: 'major',
          feature: 'minor',
          bug: 'patch',
        },
      },
    ],
    // Expose the computed version + a "published" flag as GitHub Actions job
    // outputs so the macOS build job (in the same run) can gate on them. The
    // publishCmd only runs when a release is actually published.
    [
      '@semantic-release/exec',
      {
        publishCmd:
          'echo "new_release_published=true" >> "$GITHUB_OUTPUT" && echo "new_release_version=${nextRelease.version}" >> "$GITHUB_OUTPUT"',
      },
    ],
    [
      '@semantic-release/github',
      {
        successComment: false,
        releasedLabels: false,
        // Create the release as a DRAFT; the macOS job flips it to published only
        // after the .app + appcast upload succeed (plan finding #7). A failed
        // macOS job leaves a draft, never a broken published version.
        draftRelease: true,
      },
    ],
  ],
};
