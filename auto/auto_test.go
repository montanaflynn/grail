package auto

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/anthropic"
	"github.com/montanaflynn/grail/providers/gemini"
	"github.com/montanaflynn/grail/providers/openai"
)

// clearEnv unsets every variable FromEnv reads for the duration of the test.
func clearEnv(t *testing.T) {
	t.Helper()
	t.Setenv(ProvidersEnv, "")
	for _, c := range defaultProviders {
		t.Setenv(c.env, "")
	}
}

func setAllKeys(t *testing.T) {
	t.Helper()
	t.Setenv("ANTHROPIC_API_KEY", "test-anthropic")
	t.Setenv("OPENAI_API_KEY", "test-openai")
	t.Setenv("GEMINI_API_KEY", "test-gemini")
}

func providerNames(t *testing.T) []string {
	t.Helper()
	ps, err := providersFromEnv(context.Background())
	if err != nil {
		t.Fatalf("providersFromEnv: %v", err)
	}
	names := make([]string, len(ps))
	for i, p := range ps {
		names[i] = p.Name()
	}
	return names
}

// bestTextModel returns the best text model of the client's first provider.
func bestTextModel(t *testing.T, c *grail.Client) string {
	t.Helper()
	m, err := c.GetModel(context.Background(), grail.ModelRoleText, grail.ModelTierBest)
	if err != nil {
		t.Fatalf("GetModel: %v", err)
	}
	return m.Name
}

func TestFromEnvNoneSet(t *testing.T) {
	clearEnv(t)
	t.Setenv("OPENAI_API_KEY", "   ")

	c, err := FromEnv(context.Background())
	if c != nil {
		t.Errorf("client = %v, want nil", c)
	}
	if !errors.Is(err, ErrNoProviders) {
		t.Fatalf("err = %v, want ErrNoProviders", err)
	}
	for _, env := range []string{"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"} {
		if !strings.Contains(err.Error(), env) {
			t.Errorf("err %q does not mention %s", err, env)
		}
	}
}

func TestFromEnvOneSet(t *testing.T) {
	tests := []struct {
		env  string
		name string
		best string
	}{
		{"ANTHROPIC_API_KEY", "anthropic", anthropic.ClaudeFable5_1.Name},
		{"OPENAI_API_KEY", "openai", openai.GPT6Astra.Name},
		{"GEMINI_API_KEY", "gemini", gemini.Gemini4Argon.Name},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			clearEnv(t)
			t.Setenv(tt.env, "test-key")

			if got := providerNames(t); !slices.Equal(got, []string{tt.name}) {
				t.Errorf("providers = %v, want [%s]", got, tt.name)
			}
			c, err := FromEnv(context.Background())
			if err != nil {
				t.Fatalf("FromEnv: %v", err)
			}
			if got := bestTextModel(t, c); got != tt.best {
				t.Errorf("best text model = %q, want %q", got, tt.best)
			}
		})
	}
}

func TestFromEnvAllSetDefaultOrder(t *testing.T) {
	clearEnv(t)
	setAllKeys(t)

	want := []string{"anthropic", "openai", "gemini"}
	if got := providerNames(t); !slices.Equal(got, want) {
		t.Errorf("providers = %v, want %v", got, want)
	}

	c, err := FromEnv(context.Background())
	if err != nil {
		t.Fatalf("FromEnv: %v", err)
	}
	if got := bestTextModel(t, c); got != anthropic.ClaudeFable5_1.Name {
		t.Errorf("best text model = %q, want %q", got, anthropic.ClaudeFable5_1.Name)
	}
	// Anthropic has no image models, so the first image model is OpenAI's.
	m, err := c.GetModel(context.Background(), grail.ModelRoleImage, grail.ModelTierBest)
	if err != nil {
		t.Fatalf("GetModel image: %v", err)
	}
	if m.Name != openai.GPTImage2_5Sunburst.Name {
		t.Errorf("best image model = %q, want %q", m.Name, openai.GPTImage2_5Sunburst.Name)
	}
}

func TestFromEnvOverrideOrder(t *testing.T) {
	tests := []struct {
		list string
		want []string
	}{
		{"gemini,openai,anthropic", []string{"gemini", "openai", "anthropic"}},
		{"anthropic,gemini", []string{"anthropic", "gemini"}},
		{" OpenAI , gemini,,openai ", []string{"openai", "gemini"}},
		{" , ", []string{"anthropic", "openai", "gemini"}},
	}
	for _, tt := range tests {
		t.Run(tt.list, func(t *testing.T) {
			clearEnv(t)
			setAllKeys(t)
			t.Setenv(ProvidersEnv, tt.list)

			if got := providerNames(t); !slices.Equal(got, tt.want) {
				t.Errorf("providers = %v, want %v", got, tt.want)
			}
		})
	}

	clearEnv(t)
	setAllKeys(t)
	t.Setenv(ProvidersEnv, "gemini,anthropic")
	c, err := FromEnv(context.Background())
	if err != nil {
		t.Fatalf("FromEnv: %v", err)
	}
	if got := bestTextModel(t, c); got != gemini.Gemini4Argon.Name {
		t.Errorf("best text model = %q, want %q", got, gemini.Gemini4Argon.Name)
	}
}

func TestFromEnvOverrideUnknown(t *testing.T) {
	clearEnv(t)
	setAllKeys(t)
	t.Setenv(ProvidersEnv, "openai,mistral")

	_, err := FromEnv(context.Background())
	if err == nil {
		t.Fatal("FromEnv: want error for unknown provider")
	}
	for _, s := range []string{`"mistral"`, ProvidersEnv, "anthropic, openai, gemini"} {
		if !strings.Contains(err.Error(), s) {
			t.Errorf("err %q does not mention %s", err, s)
		}
	}
}

func TestFromEnvOverrideMissingKey(t *testing.T) {
	clearEnv(t)
	t.Setenv("OPENAI_API_KEY", "test-openai")
	t.Setenv(ProvidersEnv, "openai,gemini")

	_, err := FromEnv(context.Background())
	if err == nil {
		t.Fatal("FromEnv: want error for listed provider without a key")
	}
	if !strings.Contains(err.Error(), "GEMINI_API_KEY") {
		t.Errorf("err %q does not mention GEMINI_API_KEY", err)
	}
}

func TestFromEnvPassesClientOptions(t *testing.T) {
	clearEnv(t)
	t.Setenv("OPENAI_API_KEY", "test-openai")

	// A later WithProviders option adds providers after the ones from the
	// environment.
	extra, err := gemini.New(context.Background(), gemini.WithAPIKey("test-gemini"))
	if err != nil {
		t.Fatalf("gemini.New: %v", err)
	}
	c, err := FromEnv(context.Background(), grail.WithProviders(extra))
	if err != nil {
		t.Fatalf("FromEnv: %v", err)
	}
	models, err := c.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if !slices.ContainsFunc(models, func(m grail.Model) bool { return m.Name == gemini.Gemini4Argon.Name }) {
		t.Errorf("models from WithProviders option missing: %v", models)
	}
	if got := bestTextModel(t, c); got != openai.GPT6Astra.Name {
		t.Errorf("best text model = %q, want %q", got, openai.GPT6Astra.Name)
	}
}
