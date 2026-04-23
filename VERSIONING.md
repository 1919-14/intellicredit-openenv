# Versioning Strategy

## Branches & Tags

| Name | Type | Purpose |
|------|------|---------|
| `main` | branch | Production-stable base; merges come from `v2` |
| `v1` | **tag** | Snapshot of the completed v1 (initial hackathon submission) |
| `v2` | **branch** | Active development — **all future pushes go here** |

## How It Works

```
669559a  ← first commit
   …
9b5c8b2  ← v1 tag (final v1 state)  ← also where v2 branch starts
   │
   └──► v2 branch (your future work goes here)
```

## One-Time Setup (after merging this PR)

1. **Merge this PR** into `main`.
2. Go to **Actions → "Setup Version Branches" → Run workflow**.
   - Leave the `v1_commit` field blank to tag the current `main` HEAD as `v1`.
   - The workflow will create the `v1` tag and the `v2` branch automatically.
3. **Change the default branch** to `v2`:
   - GitHub → Settings → Branches → Default branch → switch to `v2`.

## Day-to-Day Development (v2)

```bash
# Clone fresh
git clone https://github.com/1919-14/intellicredit-openenv
git checkout v2

# Work as normal — push to v2
git add .
git commit -m "feat: my v2 change"
git push origin v2        # ← this is your "next push goes to v2"
```

Pushes to `v2` automatically sync to Hugging Face Spaces (same as `main` did before).

## Accessing v1 Later

```bash
# Check out v1 snapshot (read-only)
git checkout v1

# Or create a hotfix branch off v1
git checkout -b hotfix/v1-fix v1
```
