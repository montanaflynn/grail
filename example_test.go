package grail_test

// These examples mirror the code blocks in README.md so they stay compiling.
// They use providers/mock in place of real providers, so they run without
// network access or API keys.

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/anthropic"
	"github.com/montanaflynn/grail/providers/gemini"
	"github.com/montanaflynn/grail/providers/mock"
	"github.com/montanaflynn/grail/providers/openai"
)

// fakeOpenAI stands in for the OpenAI provider. It serves images from its
// image models and fails for gpt-image-1, as the real API has since that
// model shut down on 2026-10-23.
func fakeOpenAI() *mock.CatalogProvider {
	return &mock.CatalogProvider{
		Provider: mock.Provider{
			NameVal: "openai",
			GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
				if req.Model == openai.GPTImage1.Name {
					return grail.Response{}, grail.NewGrailError(grail.InvalidArgument, "model gpt-image-1 does not exist").WithProviderName("openai")
				}
				if _, ok := grail.GetImageSpec(req.Output); ok {
					return grail.Response{
						Outputs:  []grail.OutputPart{grail.NewImageOutputPart([]byte("png"), "image/png", "")},
						Provider: grail.ProviderInfo{Models: []grail.ModelUse{{Role: "image_generation", Name: req.Model}}},
					}, nil
				}
				return grail.Response{
					Outputs:  []grail.OutputPart{grail.NewTextOutputPart("Hello from " + req.Model)},
					Provider: grail.ProviderInfo{Models: []grail.ModelUse{{Role: "language", Name: req.Model}}},
				}, nil
			},
		},
		Models: []grail.Model{
			openai.GPT6Astra, openai.GPT6Luna, openai.GPT4o,
			openai.GPTImage2_5Sunburst, openai.GPTImage2_5Flare, openai.GPTImage1,
		},
	}
}

// fakeProvider returns a provider named name that serves models and answers
// every request with a text or image naming itself.
func fakeProvider(name string, models ...grail.Model) *mock.CatalogProvider {
	return &mock.CatalogProvider{
		Provider: mock.Provider{
			NameVal: name,
			GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
				if _, ok := grail.GetImageSpec(req.Output); ok {
					return grail.Response{Outputs: []grail.OutputPart{
						grail.NewImageOutputPart([]byte("image from "+name), "image/png", ""),
					}}, nil
				}
				return grail.Response{Outputs: []grail.OutputPart{grail.NewTextOutputPart("text from " + name)}}, nil
			},
		},
		Models: models,
	}
}

// The README's model churn story: a pinned model breaks when it shuts down,
// a tier keeps working, and WithAutoMigrate rescues the pinned call.
func Example_modelChurn() {
	ctx := context.Background()
	client := grail.NewClient(fakeOpenAI())
	prompt := "A lighthouse at dawn"

	// Before: pinned to a model. Fails since gpt-image-1 shut down on 2026-10-23.
	_, err := client.Image(ctx, prompt, grail.WithModel("gpt-image-1"))
	fmt.Println("pinned:", err)

	// After: ask for a tier. Best now resolves to gpt-image-2.5-sunburst.
	png, err := client.Image(ctx, prompt, grail.WithTier(grail.Best))
	fmt.Println("tier:", len(png), "bytes", err)

	// Or keep the pin and let grail swap in the replacement.
	client = grail.NewClient(fakeOpenAI(), grail.WithAutoMigrate())
	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText(prompt)},
		Output: grail.OutputImage(grail.ImageSpec{Count: 1}),
		Model:  "gpt-image-1",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("served by:", res.Provider.Models[0].Name)
	for _, w := range res.Warnings {
		fmt.Printf("%s: %s\n", w.Code, w.Message)
	}
	// Output:
	// pinned: invalid_argument: model gpt-image-1 does not exist
	// tier: 3 bytes <nil>
	// served by: gpt-image-2.5-sunburst
	// model_migrated: gpt-image-1 shut down 2026-10-23; migrated to gpt-image-2.5-sunburst
}

// A request for a deprecated model succeeds and carries a model_deprecated
// warning. The message depends on today's date, so this example only compiles.
func Example_deprecationWarning() {
	ctx := context.Background()
	client := grail.NewClient(fakeOpenAI())

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("Summarize the release notes.")},
		Output: grail.OutputText(),
		Model:  "gpt-4o",
	})
	if err != nil {
		log.Fatal(err)
	}
	for _, w := range res.Warnings {
		fmt.Printf("%s: %s\n", w.Code, w.Message)
	}
	// Prints, until 2027-01-14:
	// model_deprecated: gpt-4o is deprecated; shuts down 2027-01-14; use gpt-6-luna
}

// The quick start's calls, against a fake provider.
func ExampleClient_Text() {
	ctx := context.Background()
	client := grail.NewClient(fakeOpenAI())

	text, err := client.Text(ctx, "Say hello.", grail.WithTier(grail.Fast))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)
	// Output: Hello from gpt-6-luna
}

func ExampleClient_Image() {
	ctx := context.Background()
	client := grail.NewClient(fakeOpenAI())

	png, err := client.Image(ctx, "A gopher surfing at sunset", grail.WithTier(grail.Fast))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(len(png), "bytes")
	// Output: 3 bytes
}

func ExampleJSON() {
	ctx := context.Background()
	provider := &mock.Provider{GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
		return grail.Response{Outputs: []grail.OutputPart{
			grail.NewJSONOutputPart([]byte(`{"name":"Tokyo","country":"Japan","population":37000000}`)),
		}}, nil
	}}
	client := grail.NewClient(provider)

	type City struct {
		Name       string `json:"name"`
		Country    string `json:"country"`
		Population int    `json:"population" jsonschema:"description=Metro area population"`
	}
	city, err := grail.JSON[City](ctx, client, "The largest city in Japan")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(city.Name, city.Country, city.Population)
	// Output: Tokyo Japan 37000000
}

// Optional fields (omitempty, pointers) stay in required but become
// nullable, as OpenAI strict mode demands.
func ExampleSchemaFor() {
	type Ticket struct {
		Title    string `json:"title"`
		Priority string `json:"priority" jsonschema:"enum=low|medium|high"`
		Assignee string `json:"assignee,omitempty" jsonschema:"description=GitHub login"`
	}
	schema, _ := json.MarshalIndent(grail.SchemaFor[Ticket](), "", "  ")
	fmt.Println(string(schema))
	// Output:
	// {
	//   "additionalProperties": false,
	//   "properties": {
	//     "assignee": {
	//       "description": "GitHub login",
	//       "type": [
	//         "string",
	//         "null"
	//       ]
	//     },
	//     "priority": {
	//       "enum": [
	//         "low",
	//         "medium",
	//         "high"
	//       ],
	//       "type": "string"
	//     },
	//     "title": {
	//       "type": "string"
	//     }
	//   },
	//   "required": [
	//     "title",
	//     "priority",
	//     "assignee"
	//   ],
	//   "type": "object"
	// }
}

// Claude for text, Gemini for images: Anthropic comes first, and image
// requests skip it because it can't generate images.
func ExampleWithProviders() {
	ctx := context.Background()
	claude := fakeProvider("anthropic", anthropic.ClaudeFable5_1, anthropic.ClaudeHaiku5_5)
	gem := fakeProvider("gemini", gemini.Gemini4Argon, gemini.Gemini3ProImage)

	client := grail.NewClient(claude, grail.WithProviders(gem))

	text, err := client.Text(ctx, "Write a tagline for a lighthouse.")
	if err != nil {
		log.Fatal(err)
	}
	img, err := client.Image(ctx, "A lighthouse at dawn")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)
	fmt.Println(string(img))
	// Output:
	// text from anthropic
	// image from gemini
}

// A retryable failure falls back to the next capable provider and records a
// fallback warning.
func ExampleWithProviders_fallback() {
	ctx := context.Background()
	claude := fakeProvider("anthropic", anthropic.ClaudeFable5_1)
	claude.GenerateFn = func(ctx context.Context, req grail.Request) (grail.Response, error) {
		return grail.Response{}, grail.NewGrailError(grail.Unavailable, "overloaded").WithProviderName("anthropic")
	}
	gem := fakeProvider("gemini", gemini.Gemini4Argon)
	client := grail.NewClient(claude, grail.WithProviders(gem))

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("Write a tagline for a lighthouse.")},
		Output: grail.OutputText(),
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("served by:", res.Provider.Name)
	for _, w := range res.Warnings {
		fmt.Printf("%s: %s\n", w.Code, w.Message)
	}
	// Output:
	// served by: gemini
	// fallback: anthropic failed (unavailable: overloaded); fell back to gemini
}

func ExampleClient_Stream() {
	ctx := context.Background()
	provider := &mock.StreamProvider{StreamFn: func(ctx context.Context, req grail.Request) iter.Seq2[grail.StreamEvent, error] {
		return mock.TextStream(grail.Usage{OutputTokens: 3}, "Once ", "upon ", "a time.")
	}}
	client := grail.NewClient(provider)

	for ev, err := range client.Stream(ctx, grail.Request{
		Inputs: []grail.Input{grail.InputText("Tell a short story.")},
		Output: grail.OutputText(),
		Tier:   grail.Fast,
	}) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(ev.Text)
		if ev.Done {
			fmt.Printf("\n(%d output tokens)\n", ev.Usage.OutputTokens)
		}
	}
	// Output:
	// Once upon a time.
	// (3 output tokens)
}

// The Request-level API from the README's Advanced section.
func ExampleClient_Generate() {
	ctx := context.Background()
	client := grail.NewClient(fakeOpenAI())

	res, err := client.Generate(ctx, grail.Request{
		Inputs: []grail.Input{
			grail.InputText("Describe this image in one sentence."),
			grail.InputImage([]byte("\x89PNG\r\n\x1a\n")),
		},
		Output: grail.OutputText(),
		Tier:   grail.Best,
	})
	if err != nil {
		log.Fatal(err)
	}
	text, _ := res.Text()
	fmt.Println(res.Provider.Name+":", text)
	// Output: openai: Hello from gpt-6-astra
}
