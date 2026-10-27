package main

import (
	"fmt"
	"io"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/montanaflynn/grail"
)

// liveModel is one model as reported by a provider's list API.
type liveModel struct {
	ID      string
	Created time.Time // Release time; zero when the API does not report one
	Actions []string  // Supported methods (Gemini only); nil when not reported
	Retired bool      // Listed for the record but no longer callable (Anthropic)
}

// diffResult is the drift between grail's catalog and a provider's live list.
type diffResult struct {
	// Missing lists known models that are absent from the API and not already
	// recorded as shut down in grail. A missing model was most likely shut down.
	// Models listed in MissingTiers are not repeated here.
	Missing []string
	// MissingTiers lists default and best/fast tier models absent from the API,
	// regardless of their deprecation records.
	MissingTiers []string
	// New lists text and image generation models the API serves that grail
	// does not know about. See isCandidate for the filter.
	New []string
}

// Breaking reports whether callers relying on grail's catalog would see
// failures: a known model or a default/tier model is gone.
func (d diffResult) Breaking() bool {
	return len(d.Missing) > 0 || len(d.MissingTiers) > 0
}

// Empty reports whether there is no drift at all.
func (d diffResult) Empty() bool {
	return !d.Breaking() && len(d.New) == 0
}

var (
	// snapshotSuffix matches a dated or numbered snapshot suffix such as
	// -2025-08-07, -20251101, -0613 or -001.
	snapshotSuffix = regexp.MustCompile(`-(\d{4}-\d{2}-\d{2}|\d{8}|\d{3,4})$`)
	// familyVersion matches a version directly after the family name, e.g.
	// 3.8 in gemini-3.8-flash. Named models such as gemini-nano-banana-2.1
	// have no family version.
	familyVersion = regexp.MustCompile(`^[a-z]+-(\d+(?:\.\d+)?)(?:-|$)`)
)

// excludedTerms mark model families that do not generate text or images:
// embeddings, speech and audio, realtime, moderation, search, video and
// open-weight side families grail does not route to.
var excludedTerms = []string{
	"embed", "tts", "whisper", "transcribe", "audio", "speech", "realtime",
	"live", "moderation", "search", "computer-use", "sora", "veo", "gemma",
	"aqa", "learnlm",
}

// diffModels compares grail's known catalog with a provider's live models.
// tiers holds the default and best/fast tier model IDs, which must be live.
// Known models already shut down as of now are expected to be gone and are not
// reported as missing.
func diffModels(known []grail.Model, tiers []string, live []liveModel, now time.Time) diffResult {
	liveIDs := map[string]bool{}
	for _, m := range live {
		if !m.Retired {
			liveIDs[m.ID] = true
		}
	}
	knownIDs := map[string]bool{}
	for _, m := range known {
		knownIDs[m.Name] = true
	}

	// present reports whether id is served directly or through a snapshot,
	// e.g. the alias claude-opus-4-5 listed only as claude-opus-4-5-20251101.
	present := func(id string) bool {
		if liveIDs[id] {
			return true
		}
		for liveID := range liveIDs {
			if snapshotBase(liveID) == id {
				return true
			}
		}
		return false
	}

	var d diffResult
	for _, id := range tiers {
		if id != "" && !present(id) && !slices.Contains(d.MissingTiers, id) {
			d.MissingTiers = append(d.MissingTiers, id)
		}
	}
	for _, m := range known {
		if !m.IsShutDown(now) && !present(m.Name) && !slices.Contains(d.MissingTiers, m.Name) {
			d.Missing = append(d.Missing, m.Name)
		}
	}

	cutoff := newestKnownRelease(live, knownIDs)
	minVersion, hasMinVersion := lowestVersion(tiers)
	for _, m := range live {
		if knownIDs[m.ID] || !isCandidate(m, liveIDs, knownIDs) {
			continue
		}
		// Recency: anything released before grail's newest known model existed
		// at the last catalog update and was deliberately left out.
		if !cutoff.IsZero() && !m.Created.IsZero() && m.Created.Before(cutoff) {
			continue
		}
		// Without release dates, skip generations older than the oldest
		// default/tier model.
		if cutoff.IsZero() && hasMinVersion {
			if v, ok := modelVersion(m.ID); ok && v < minVersion {
				continue
			}
		}
		d.New = append(d.New, m.ID)
	}

	slices.Sort(d.Missing)
	slices.Sort(d.MissingTiers)
	slices.Sort(d.New)
	d.New = slices.Compact(d.New)
	return d
}

// isCandidate reports whether an unknown live model is a text or image
// generation model worth adding to grail. It drops retired models, fine-tunes,
// "-latest" aliases, non-generation families (see excludedTerms), Gemini
// models without generateContent, and snapshots of an ID that is already live
// or known.
func isCandidate(m liveModel, liveIDs, knownIDs map[string]bool) bool {
	id := strings.ToLower(m.ID)
	if m.Retired || strings.HasPrefix(id, "ft:") || strings.HasSuffix(id, "-latest") {
		return false
	}
	for _, term := range excludedTerms {
		if strings.Contains(id, term) {
			return false
		}
	}
	if m.Actions != nil && !slices.Contains(m.Actions, "generateContent") {
		return false
	}
	if base := snapshotBase(m.ID); base != m.ID && (liveIDs[base] || knownIDs[base]) {
		return false
	}
	return true
}

// snapshotBase strips a dated or numbered snapshot suffix from id.
func snapshotBase(id string) string {
	return snapshotSuffix.ReplaceAllString(id, "")
}

// newestKnownRelease returns the latest release time among live models grail
// knows, or zero when the API reports no release times.
func newestKnownRelease(live []liveModel, knownIDs map[string]bool) time.Time {
	var newest time.Time
	for _, m := range live {
		if (knownIDs[m.ID] || knownIDs[snapshotBase(m.ID)]) && m.Created.After(newest) {
			newest = m.Created
		}
	}
	return newest
}

// lowestVersion returns the lowest parsable version among ids.
func lowestVersion(ids []string) (float64, bool) {
	var lowest float64
	var found bool
	for _, id := range ids {
		if v, ok := modelVersion(id); ok && (!found || v < lowest) {
			lowest, found = v, true
		}
	}
	return lowest, found
}

// modelVersion parses the family version of id, e.g. 3.8 from
// gemini-3.8-flash. Snapshot suffixes are ignored.
func modelVersion(id string) (float64, bool) {
	match := familyVersion.FindStringSubmatch(snapshotBase(id))
	if match == nil {
		return 0, false
	}
	v, err := strconv.ParseFloat(match[1], 64)
	return v, err == nil
}

// writeDiff writes a Markdown section for one provider's diff. Each drifted
// model is a "- " list item, so callers can detect drift by that prefix.
func writeDiff(w io.Writer, provider string, d diffResult) {
	fmt.Fprintf(w, "## %s\n\n", provider)
	if d.Empty() {
		fmt.Fprintln(w, "No drift.")
		fmt.Fprintln(w)
		return
	}
	section := func(title string, ids []string) {
		if len(ids) == 0 {
			return
		}
		fmt.Fprintf(w, "%s\n\n", title)
		for _, id := range ids {
			fmt.Fprintf(w, "- `%s`\n", id)
		}
		fmt.Fprintln(w)
	}
	section("**Missing default/tier models** (grail selects these; requests will fail):", d.MissingTiers)
	section("**Missing** (known to grail, gone from the API; likely shut down):", d.Missing)
	section("**New** (served by the API, unknown to grail; text and image generation only):", d.New)
}
