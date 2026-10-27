package main

import (
	"bytes"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/montanaflynn/grail"
)

var testNow = time.Date(2026, time.October, 27, 0, 0, 0, 0, time.UTC)

func day(year int, month time.Month, d int) time.Time {
	return time.Date(year, month, d, 0, 0, 0, 0, time.UTC)
}

func ids(names ...string) []liveModel {
	live := make([]liveModel, len(names))
	for i, n := range names {
		live[i] = liveModel{ID: n}
	}
	return live
}

func models(names ...string) []grail.Model {
	known := make([]grail.Model, len(names))
	for i, n := range names {
		known[i] = grail.Model{Name: n}
	}
	return known
}

func TestDiffModelsNoDrift(t *testing.T) {
	d := diffModels(models("a-1", "a-2"), []string{"a-1"}, ids("a-1", "a-2"), nil, testNow)
	if !d.Empty() || d.Breaking() {
		t.Fatalf("expected no drift, got %+v", d)
	}
}

func TestDiffModelsMissing(t *testing.T) {
	d := diffModels(models("gpt-6-astra", "gpt-5.2"), nil, ids("gpt-6-astra"), nil, testNow)
	if !slices.Equal(d.Missing, []string{"gpt-5.2"}) {
		t.Fatalf("Missing = %v, want [gpt-5.2]", d.Missing)
	}
	if !d.Breaking() {
		t.Fatal("expected missing model to be breaking")
	}
}

func TestDiffModelsShutDownModelIsNotMissing(t *testing.T) {
	known := []grail.Model{
		{Name: "gpt-image-2.5-sunburst"},
		{Name: "gpt-image-1", Deprecation: &grail.Deprecation{Shutdown: day(2026, time.October, 23)}},
		// Deprecated with a future shutdown: disappearing early is drift.
		{Name: "gpt-image-1-mini", Deprecation: &grail.Deprecation{Shutdown: day(2026, time.December, 1)}},
	}
	d := diffModels(known, nil, ids("gpt-image-2.5-sunburst"), nil, testNow)
	if !slices.Equal(d.Missing, []string{"gpt-image-1-mini"}) {
		t.Fatalf("Missing = %v, want [gpt-image-1-mini]", d.Missing)
	}
}

func TestDiffModelsMissingTier(t *testing.T) {
	// A shut-down default is still reported: grail would select a dead model.
	known := []grail.Model{
		{Name: "gemini-4-argon", Deprecation: &grail.Deprecation{Shutdown: day(2026, time.October, 1)}},
		{Name: "gemini-3.8-flash"},
	}
	tiers := []string{"gemini-4-argon", "gemini-4-argon", "gemini-3.8-flash", ""}
	d := diffModels(known, tiers, ids("gemini-3.8-flash"), nil, testNow)
	if !slices.Equal(d.MissingTiers, []string{"gemini-4-argon"}) {
		t.Fatalf("MissingTiers = %v, want [gemini-4-argon]", d.MissingTiers)
	}
	if len(d.Missing) != 0 {
		t.Fatalf("Missing = %v, want tier models listed only under MissingTiers", d.Missing)
	}
	if !d.Breaking() {
		t.Fatal("expected missing tier model to be breaking")
	}
}

func TestDiffModelsAliasServedBySnapshot(t *testing.T) {
	d := diffModels(models("claude-opus-4-5"), []string{"claude-opus-4-5"}, ids("claude-opus-4-5-20251101"), nil, testNow)
	if !d.Empty() {
		t.Fatalf("expected alias to match its dated snapshot, got %+v", d)
	}
}

func TestDiffModelsRetiredCountsAsMissing(t *testing.T) {
	live := []liveModel{
		{ID: "claude-fable-5-1"},
		{ID: "claude-sonnet-4-5", Retired: true},
		{ID: "claude-sonnet-3-7", Retired: true},
	}
	d := diffModels(models("claude-fable-5-1", "claude-sonnet-4-5"), nil, live, nil, testNow)
	if !slices.Equal(d.Missing, []string{"claude-sonnet-4-5"}) {
		t.Fatalf("Missing = %v, want [claude-sonnet-4-5]", d.Missing)
	}
	if len(d.New) != 0 {
		t.Fatalf("New = %v, want retired models excluded", d.New)
	}
}

func TestDiffModelsNewFilter(t *testing.T) {
	live := ids(
		"gpt-6-astra",
		"gpt-6.2-nova",                  // new text model
		"gpt-image-3",                   // new image model
		"gpt-6.2-nova-2026-10-20",       // snapshot of a live alias
		"gpt-6-astra-2026-09-01",        // snapshot of a known model
		"text-embedding-4-large",        // embeddings
		"gpt-6-mini-tts",                // speech
		"gpt-6-audio-preview",           // audio
		"gpt-6-realtime",                // realtime
		"whisper-2",                     // transcription
		"gpt-6-transcribe",              // transcription
		"omni-moderation-2026-10-01",    // moderation
		"gpt-6-search-preview",          // search
		"sora-3",                        // video
		"ft:gpt-6-luna:acme:custom:abc", // fine-tune
		"chatgpt-6-latest",              // moving alias
	)
	d := diffModels(models("gpt-6-astra"), nil, live, nil, testNow)
	want := []string{"gpt-6.2-nova", "gpt-image-3"}
	if !slices.Equal(d.New, want) {
		t.Fatalf("New = %v, want %v", d.New, want)
	}
	if d.Breaking() {
		t.Fatal("new models alone must not be breaking")
	}
}

func TestDiffModelsGeminiActions(t *testing.T) {
	live := []liveModel{
		{ID: "gemini-3.8-flash", Actions: []string{"generateContent"}},
		{ID: "gemini-4.1-argon", Actions: []string{"generateContent", "countTokens"}},
		{ID: "imagen-5.0-generate", Actions: []string{"predict"}},
		{ID: "gemini-4-embedding", Actions: []string{"embedContent"}},
		{ID: "gemini-4-flash-live", Actions: []string{"bidiGenerateContent"}},
		{ID: "gemma-4-27b-it", Actions: []string{"generateContent"}},
		{ID: "gemini-3.8-flash-001", Actions: []string{"generateContent"}},
	}
	d := diffModels(models("gemini-3.8-flash"), []string{"gemini-3.8-flash"}, live, nil, testNow)
	if !slices.Equal(d.New, []string{"gemini-4.1-argon"}) {
		t.Fatalf("New = %v, want [gemini-4.1-argon]", d.New)
	}
}

func TestDiffModelsRecencyByReleaseDate(t *testing.T) {
	live := []liveModel{
		{ID: "gpt-6-astra", Created: day(2026, time.September, 1)},
		{ID: "gpt-4.1", Created: day(2025, time.April, 14)},         // legacy, deliberately left out
		{ID: "gpt-6-astra-pro", Created: day(2026, time.August, 1)}, // predates the newest known model
		{ID: "gpt-6.2-nova", Created: day(2026, time.October, 20)},
		{ID: "gpt-6.3-nova"}, // unknown release date is kept
	}
	d := diffModels(models("gpt-6-astra"), nil, live, nil, testNow)
	want := []string{"gpt-6.2-nova", "gpt-6.3-nova"}
	if !slices.Equal(d.New, want) {
		t.Fatalf("New = %v, want %v", d.New, want)
	}
}

func TestDiffModelsRecencyByVersion(t *testing.T) {
	// No release dates: generations older than the oldest tier model are skipped.
	live := ids("gemini-3-pro-image", "gemini-4-argon", "gemini-2.5-pro", "gemini-3.9-flash", "gemini-nano-banana-3")
	tiers := []string{"gemini-4-argon", "gemini-3-pro-image", "gemini-nano-banana-2.1"}
	d := diffModels(models(tiers...), tiers, live, nil, testNow)
	if !slices.Equal(d.MissingTiers, []string{"gemini-nano-banana-2.1"}) {
		t.Fatalf("MissingTiers = %v", d.MissingTiers)
	}
	want := []string{"gemini-3.9-flash", "gemini-nano-banana-3"}
	if !slices.Equal(d.New, want) {
		t.Fatalf("New = %v, want %v", d.New, want)
	}
}

func TestSnapshotBase(t *testing.T) {
	tests := map[string]string{
		"gpt-4o-2024-08-06":           "gpt-4o",
		"claude-opus-4-5-20251101":    "claude-opus-4-5",
		"gpt-4-0613":                  "gpt-4",
		"gemini-2.0-flash-001":        "gemini-2.0-flash",
		"claude-opus-4-5":             "claude-opus-4-5",
		"gemini-nano-banana-2.1":      "gemini-nano-banana-2.1",
		"gpt-image-2.5-sunburst":      "gpt-image-2.5-sunburst",
		"gemini-3.1-flash-lite-image": "gemini-3.1-flash-lite-image",
	}
	for in, want := range tests {
		if got := snapshotBase(in); got != want {
			t.Errorf("snapshotBase(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestWriteDiff(t *testing.T) {
	var buf bytes.Buffer
	writeDiff(&buf, "openai", diffResult{})
	if got := buf.String(); !strings.Contains(got, "## openai") || !strings.Contains(got, "No drift.") || strings.Contains(got, "\n- ") {
		t.Fatalf("clean diff output = %q", got)
	}

	buf.Reset()
	writeDiff(&buf, "gemini", diffResult{
		Missing:      []string{"gemini-3.6-flash"},
		MissingTiers: []string{"gemini-4-argon"},
		New:          []string{"gemini-4.1-argon"},
	})
	got := buf.String()
	for _, want := range []string{"## gemini", "- `gemini-3.6-flash`", "- `gemini-4-argon`", "- `gemini-4.1-argon`", "Missing default/tier", "New"} {
		if !strings.Contains(got, want) {
			t.Errorf("output missing %q:\n%s", want, got)
		}
	}
}

func TestCatalogTiersAreKnown(t *testing.T) {
	// Constructing providers makes no API calls, so a dummy key is enough.
	for _, name := range []string{"openai", "gemini", "anthropic"} {
		known, tiers, err := catalog(t.Context(), name, "test-key")
		if err != nil {
			t.Fatalf("catalog(%s): %v", name, err)
		}
		if len(known) == 0 || len(tiers) == 0 {
			t.Fatalf("catalog(%s): known=%d tiers=%d", name, len(known), len(tiers))
		}
		// With every known model live, nothing may be reported as drift.
		var live []liveModel
		for _, m := range known {
			live = append(live, liveModel{ID: m.Name})
		}
		if d := diffModels(known, tiers, live, nil, testNow); !d.Empty() {
			t.Errorf("catalog(%s): tier models not in KnownModels: %+v", name, d)
		}
	}
}

func TestParseProviders(t *testing.T) {
	got, err := parseProviders("all")
	if err != nil || !slices.Equal(got, []string{"openai", "gemini", "anthropic"}) {
		t.Fatalf("parseProviders(all) = %v, %v", got, err)
	}
	got, err = parseProviders(" Anthropic, openai")
	if err != nil || !slices.Equal(got, []string{"anthropic", "openai"}) {
		t.Fatalf("parseProviders(list) = %v, %v", got, err)
	}
	if _, err := parseProviders("openai,modelslab"); err == nil {
		t.Fatal("expected error for unknown provider")
	}
}

func TestDiffModelsTightenedFilter(t *testing.T) {
	// The live Gemini "new" list from 2026-10-27, before acknowledgement.
	act := []string{"generateContent"}
	var live []liveModel
	for _, id := range []string{
		"gemini-3.8-flash", "gemini-3.1-flash-lite",
		"antigravity-preview-05-2026", "antigravity-preview-09-2026",
		"gemini-3.1-flash-lite-preview", "gemini-3.1-pro-preview-customtools",
		"gemini-3.7-flash", "gemini-omni-1.1-flash", "gemini-omni-flash-preview",
		"gemini-robotics-er-2-preview", "lyria-3-clip-preview", "lyria-3-pro-preview",
		"lyria-3.5", "nano-banana-pro-preview",
	} {
		live = append(live, liveModel{ID: id, Actions: act})
	}
	known := models("gemini-3.8-flash", "gemini-3.1-flash-lite", "gemini-3-pro-image")
	tiers := []string{"gemini-3.8-flash", "gemini-3-pro-image"}

	d := diffModels(known, tiers, live, nil, testNow)
	want := []string{"gemini-3.7-flash", "gemini-omni-1.1-flash", "gemini-omni-flash-preview", "nano-banana-pro-preview"}
	if !slices.Equal(d.New, want) {
		t.Fatalf("New = %v, want %v", d.New, want)
	}
}

func TestDiffModelsPreviewOfLiveModel(t *testing.T) {
	d := diffModels(models("gpt-6-astra"), nil, ids("gpt-6-astra", "gpt-6-astra-preview", "gpt-7-preview"), nil, testNow)
	if !slices.Equal(d.New, []string{"gpt-7-preview"}) {
		t.Fatalf("New = %v, want [gpt-7-preview]", d.New)
	}
}

func TestDiffModelsAcknowledged(t *testing.T) {
	live := ids("gemini-3.8-flash", "gemini-3.7-flash", "gemini-4.1-argon")
	acked := map[string]bool{"gemini-3.7-flash": true, "gemini-3.8-flash": true}
	d := diffModels(models("gemini-3.8-flash"), nil, live, acked, testNow)
	if !slices.Equal(d.New, []string{"gemini-4.1-argon"}) {
		t.Fatalf("New = %v, want [gemini-4.1-argon]", d.New)
	}
	// Acknowledgement affects only "new"; a known model still goes missing.
	d = diffModels(models("gemini-3.8-flash"), nil, ids("gemini-3.7-flash"), acked, testNow)
	if !slices.Equal(d.Missing, []string{"gemini-3.8-flash"}) {
		t.Fatalf("Missing = %v, want [gemini-3.8-flash]", d.Missing)
	}
}

func TestParseAcknowledged(t *testing.T) {
	in := "# header comment\n\n" +
		"gemini/gemini-3.7-flash   # deprecated\n" +
		"  gemini/nano-banana-pro-preview\n" +
		"openai/gpt-6-astra-pro#no space before comment\n"
	got, err := parseAcknowledged(strings.NewReader(in))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]map[string]bool{
		"gemini": {"gemini-3.7-flash": true, "nano-banana-pro-preview": true},
		"openai": {"gpt-6-astra-pro": true},
	}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for p, idset := range want {
		if len(got[p]) != len(idset) {
			t.Fatalf("got[%s] = %v, want %v", p, got[p], idset)
		}
		for id := range idset {
			if !got[p][id] {
				t.Errorf("missing %s/%s", p, id)
			}
		}
	}

	for _, bad := range []string{"gemini-3.7-flash", "/gemini-3.7-flash", "gemini/", "gemini/a b"} {
		if _, err := parseAcknowledged(strings.NewReader(bad + "\n")); err == nil {
			t.Errorf("parseAcknowledged(%q): expected error", bad)
		}
	}
}

func TestAcknowledgedFileParses(t *testing.T) {
	acked, err := loadAcknowledged("../../acknowledged.txt")
	if err != nil {
		t.Fatal(err)
	}
	if len(acked) == 0 {
		t.Fatal("acknowledged.txt: no entries")
	}
	for p := range acked {
		if envFor(p) == "" {
			t.Errorf("acknowledged.txt: unknown provider %q", p)
		}
	}
}
