// Command listmodels prints model IDs from provider APIs for auditing grail defaults.
//
// With -diff it compares the live IDs against each provider's KnownModels()
// and prints the drift as Markdown. Exit codes:
//
//	0  success; in -diff mode, no missing models (new models alone are not an error)
//	1  -diff found missing known models or missing default/tier models
//	2  usage error
//	3  a provider could not be listed (missing API key or API error)
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/montanaflynn/grail"
	grailanthropic "github.com/montanaflynn/grail/providers/anthropic"
	grailgemini "github.com/montanaflynn/grail/providers/gemini"
	grailopenai "github.com/montanaflynn/grail/providers/openai"

	"github.com/anthropics/anthropic-sdk-go/v2"
	anthropicoption "github.com/anthropics/anthropic-sdk-go/v2/option"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"google.golang.org/genai"
)

const (
	exitDrift = 1
	exitUsage = 2
	exitError = 3
)

// providers maps each provider name to its API key environment variable.
var providers = []struct{ name, env string }{
	{"openai", "OPENAI_API_KEY"},
	{"gemini", "GEMINI_API_KEY"},
	{"anthropic", "ANTHROPIC_API_KEY"},
}

func main() {
	provider := flag.String("provider", "all", "Provider(s) to list, comma-separated: openai, gemini, anthropic, or all")
	diff := flag.Bool("diff", false, "Compare live IDs against grail's KnownModels() and print missing and new models")
	flag.Parse()

	names, err := parseProviders(*provider)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintf(os.Stderr, "Examples:\n")
		fmt.Fprintf(os.Stderr, "  listmodels -provider openai\n")
		fmt.Fprintf(os.Stderr, "  listmodels -provider gemini,anthropic\n")
		fmt.Fprintf(os.Stderr, "  listmodels -provider all -diff\n")
		os.Exit(exitUsage)
	}

	ctx := context.Background()
	var failed, drifted bool
	for _, name := range names {
		env := envFor(name)
		key := strings.TrimSpace(os.Getenv(env))
		if key == "" {
			fmt.Fprintf(os.Stderr, "Error: %s is not set\n", env)
			fmt.Fprintf(os.Stderr, "Example: %s=... listmodels -provider %s\n", env, name)
			failed = true
			continue
		}

		live, err := listLive(ctx, name, key)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error listing %s models: %v\n", name, err)
			failed = true
			continue
		}

		if !*diff {
			fmt.Printf("# %s\n", name)
			for _, m := range live {
				fmt.Println(m.ID)
			}
			continue
		}

		known, tiers, err := catalog(ctx, name, key)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error loading grail %s catalog: %v\n", name, err)
			failed = true
			continue
		}
		d := diffModels(known, tiers, live, time.Now())
		writeDiff(os.Stdout, name, d)
		drifted = drifted || d.Breaking()
	}

	switch {
	case failed:
		os.Exit(exitError)
	case drifted:
		os.Exit(exitDrift)
	}
}

// parseProviders splits a comma-separated provider list, expanding "all".
func parseProviders(s string) ([]string, error) {
	var names []string
	for _, part := range strings.Split(strings.ToLower(s), ",") {
		part = strings.TrimSpace(part)
		switch {
		case part == "all":
			for _, p := range providers {
				names = append(names, p.name)
			}
		case envFor(part) != "":
			names = append(names, part)
		default:
			return nil, fmt.Errorf("unknown provider %q", part)
		}
	}
	return names, nil
}

func envFor(name string) string {
	for _, p := range providers {
		if p.name == name {
			return p.env
		}
	}
	return ""
}

// listLive returns the provider's live models sorted by ID.
func listLive(ctx context.Context, name, key string) ([]liveModel, error) {
	var live []liveModel
	var err error
	switch name {
	case "openai":
		live, err = listOpenAI(ctx, key)
	case "gemini":
		live, err = listGemini(ctx, key)
	case "anthropic":
		live, err = listAnthropic(ctx, key)
	}
	sort.Slice(live, func(i, j int) bool { return live[i].ID < live[j].ID })
	return live, err
}

func listOpenAI(ctx context.Context, key string) ([]liveModel, error) {
	client := openai.NewClient(option.WithAPIKey(key))
	page, err := client.Models.List(ctx)
	if err != nil {
		return nil, err
	}

	live := make([]liveModel, 0, len(page.Data))
	for _, m := range page.Data {
		if m.ID != "" {
			live = append(live, liveModel{ID: m.ID, Created: unixTime(m.Created)})
		}
	}
	return live, nil
}

func listGemini(ctx context.Context, key string) ([]liveModel, error) {
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  key,
	})
	if err != nil {
		return nil, fmt.Errorf("create client: %w", err)
	}

	var live []liveModel
	for model, err := range client.Models.All(ctx) {
		if err != nil {
			return nil, err
		}
		// model.Name is like "models/gemini-3.8-flash"; grail uses the bare ID.
		if id := strings.TrimPrefix(model.Name, "models/"); id != "" {
			actions := model.SupportedActions
			if actions == nil {
				actions = []string{}
			}
			live = append(live, liveModel{ID: id, Actions: actions})
		}
	}
	return live, nil
}

func listAnthropic(ctx context.Context, key string) ([]liveModel, error) {
	client := anthropic.NewClient(anthropicoption.WithAPIKey(key))
	pager := client.Models.ListAutoPaging(ctx, anthropic.ModelListParams{Limit: anthropic.Int(1000)})

	var live []liveModel
	for pager.Next() {
		m := pager.Current()
		if m.ID != "" {
			live = append(live, liveModel{
				ID:      m.ID,
				Created: unixTime(m.CreatedAt.Unix()),
				Retired: m.Lifecycle == anthropic.ModelInfoLifecycleRetired,
			})
		}
	}
	return live, pager.Err()
}

// unixTime converts seconds to a time, treating non-positive values (unknown
// release dates) as zero.
func unixTime(sec int64) time.Time {
	if sec <= 0 {
		return time.Time{}
	}
	return time.Unix(sec, 0).UTC()
}

// catalog returns grail's known models for a provider and the IDs grail
// selects by default or by tier. Constructing a provider makes no API calls.
func catalog(ctx context.Context, name, key string) ([]grail.Model, []string, error) {
	var p interface {
		grail.KnownModelLister
		grail.ModelCatalog
	}
	var defaults []string
	var err error
	switch name {
	case "openai":
		p, err = grailopenai.New(grailopenai.WithAPIKey(key))
		defaults = []string{grailopenai.DefaultTextModelName, grailopenai.DefaultImageModelName}
	case "gemini":
		p, err = grailgemini.New(ctx, grailgemini.WithAPIKey(key))
		defaults = []string{grailgemini.DefaultTextModelName, grailgemini.DefaultImageModelName}
	case "anthropic":
		p, err = grailanthropic.New(grailanthropic.WithAPIKey(key))
		defaults = []string{grailanthropic.DefaultTextModelName}
	default:
		err = errors.New("no grail provider")
	}
	if err != nil {
		return nil, nil, err
	}

	tiers := append(defaults,
		p.BestTextModel().Name, p.FastTextModel().Name,
		p.BestImageModel().Name, p.FastImageModel().Name,
	)
	return p.KnownModels(), tiers, nil
}
