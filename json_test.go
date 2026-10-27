package grail_test

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/montanaflynn/grail"
	"github.com/montanaflynn/grail/providers/mock"
)

type schemaInner struct {
	Label string `json:"label"`
}

type schemaEmbedded struct {
	Shared string `json:"shared"`
	Hidden string `json:"name"` // hidden by the outer "name" field
}

type schemaOuter struct {
	schemaEmbedded
	Name   string      `json:"name"`
	Inner  schemaInner `json:"inner"`
	Plain  int
	Skip   string `json:"-"`
	hidden string // unexported, skipped
}

type schemaOptional struct {
	Nick  string       `json:"nick,omitempty"`
	Count *int         `json:"count"`
	Inner *schemaInner `json:"inner,omitempty"`
	Zero  time.Time    `json:"zero,omitzero"`
}

type schemaTagged struct {
	Status string   `json:"status" jsonschema:"description=Current state, if known,enum=open|closed"`
	Level  int      `json:"level" jsonschema:"enum=1|2|3"`
	Tags   []string `json:"tags" jsonschema:"enum=a|b"`
	Maybe  *string  `json:"maybe" jsonschema:"enum=x|y"`
	ID     int64    `json:"id,string"`
}

type schemaNode struct {
	Children []schemaNode `json:"children"`
}

func TestSchemaFor(t *testing.T) {
	tests := []struct {
		name   string
		schema func() map[string]any
		want   string
	}{
		{"string", grail.SchemaFor[string], `{"type":"string"}`},
		{"bool", grail.SchemaFor[bool], `{"type":"boolean"}`},
		{"int", grail.SchemaFor[int], `{"type":"integer"}`},
		{"uint8", grail.SchemaFor[uint8], `{"type":"integer"}`},
		{"float", grail.SchemaFor[float64], `{"type":"number"}`},
		{"time", grail.SchemaFor[time.Time], `{"type":"string","format":"date-time"}`},
		{"bytes", grail.SchemaFor[[]byte], `{"type":"string"}`},
		{"slice", grail.SchemaFor[[]int], `{"type":"array","items":{"type":"integer"}}`},
		{"array", grail.SchemaFor[[2]string], `{"type":"array","items":{"type":"string"}}`},
		{"slice of pointers", grail.SchemaFor[[]*string], `{"type":"array","items":{"type":["string","null"]}}`},
		{"map", grail.SchemaFor[map[string]float64], `{"type":"object","additionalProperties":{"type":"number"}}`},
		{"root pointer", grail.SchemaFor[*schemaInner], `{
			"type":"object",
			"properties":{"label":{"type":"string"}},
			"required":["label"],
			"additionalProperties":false
		}`},
		{"struct", grail.SchemaFor[schemaOuter], `{
			"type":"object",
			"properties":{
				"shared":{"type":"string"},
				"name":{"type":"string"},
				"inner":{
					"type":"object",
					"properties":{"label":{"type":"string"}},
					"required":["label"],
					"additionalProperties":false
				},
				"Plain":{"type":"integer"}
			},
			"required":["shared","name","inner","Plain"],
			"additionalProperties":false
		}`},
		{"optional fields", grail.SchemaFor[schemaOptional], `{
			"type":"object",
			"properties":{
				"nick":{"type":["string","null"]},
				"count":{"type":["integer","null"]},
				"inner":{
					"type":["object","null"],
					"properties":{"label":{"type":"string"}},
					"required":["label"],
					"additionalProperties":false
				},
				"zero":{"type":["string","null"],"format":"date-time"}
			},
			"required":["nick","count","inner","zero"],
			"additionalProperties":false
		}`},
		{"jsonschema tags", grail.SchemaFor[schemaTagged], `{
			"type":"object",
			"properties":{
				"status":{"type":"string","description":"Current state, if known","enum":["open","closed"]},
				"level":{"type":"integer","enum":[1,2,3]},
				"tags":{"type":"array","items":{"type":"string","enum":["a","b"]}},
				"maybe":{"type":["string","null"],"enum":["x","y",null]},
				"id":{"type":"string"}
			},
			"required":["status","level","tags","maybe","id"],
			"additionalProperties":false
		}`},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assertJSONEqual(t, tt.schema(), tt.want)
		})
	}
}

// SchemaExported is exported so encoding/json promotes it through a pointer.
type SchemaExported struct {
	Label string `json:"label"`
}

func TestSchemaForEmbeddedPointer(t *testing.T) {
	type withPtr struct {
		*SchemaExported
		*schemaInner        // unexported embedded pointer, ignored like encoding/json
		ID           string `json:"id"`
	}
	assertJSONEqual(t, grail.SchemaFor[withPtr](), `{
		"type":"object",
		"properties":{"label":{"type":["string","null"]},"id":{"type":"string"}},
		"required":["label","id"],
		"additionalProperties":false
	}`)
}

func TestSchemaForUnsupported(t *testing.T) {
	tests := []struct {
		name   string
		schema func() map[string]any
		want   string
	}{
		{"recursive", grail.SchemaFor[schemaNode], "recursive type"},
		{"interface", grail.SchemaFor[any], "not supported"},
		{"chan", grail.SchemaFor[chan int], "not supported"},
		{"int map key", grail.SchemaFor[map[int]string], "only string keys"},
		{"bad enum", grail.SchemaFor[struct {
			N int `json:"n" jsonschema:"enum=one"`
		}], "not a valid integer"},
		{"unknown tag key", grail.SchemaFor[struct {
			N int `json:"n" jsonschema:"minimum=1"`
		}], "unknown jsonschema tag key"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				msg, _ := r.(string)
				if !strings.Contains(msg, tt.want) {
					t.Fatalf("panic = %v, want message containing %q", r, tt.want)
				}
			}()
			tt.schema()
		})
	}
}

type recipe struct {
	Name        string    `json:"name" jsonschema:"description=Dish name"`
	Servings    int       `json:"servings"`
	Ingredients []string  `json:"ingredients"`
	Notes       string    `json:"notes,omitempty"`
	Created     time.Time `json:"created"`
}

func TestJSON(t *testing.T) {
	ctx := context.Background()
	var got grail.Request
	prov := &mock.Provider{
		GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
			got = req
			return grail.Response{
				Outputs: []grail.OutputPart{grail.NewJSONOutputPart([]byte(`{
					"name":"Pancakes","servings":4,"ingredients":["flour","milk"],
					"notes":null,"created":"2026-10-27T09:00:00Z"
				}`))},
			}, nil
		},
	}
	client := grail.NewClient(prov)

	r, err := grail.JSON[recipe](ctx, client, "a pancake recipe", grail.WithTier(grail.Fast))
	if err != nil {
		t.Fatalf("JSON: %v", err)
	}
	want := recipe{
		Name:        "Pancakes",
		Servings:    4,
		Ingredients: []string{"flour", "milk"},
		Created:     time.Date(2026, 10, 27, 9, 0, 0, 0, time.UTC),
	}
	if !reflect.DeepEqual(r, want) {
		t.Fatalf("recipe = %+v, want %+v", r, want)
	}

	schema, strict, ok := grail.GetJSONOutput(got.Output)
	if !ok {
		t.Fatalf("expected JSON output, got %T", got.Output)
	}
	if !strict {
		t.Fatal("expected strict JSON output")
	}
	if !reflect.DeepEqual(schema, grail.SchemaFor[recipe]()) {
		t.Fatalf("schema = %v, want SchemaFor[recipe]", schema)
	}
	if s, ok := grail.AsTextInput(got.Inputs[0]); !ok || s != "a pancake recipe" {
		t.Fatalf("first input = %q (text=%v), want prompt", s, ok)
	}
	if got.Tier != grail.Fast {
		t.Fatalf("tier = %q, want %q", got.Tier, grail.Fast)
	}
}

func TestJSONErrors(t *testing.T) {
	ctx := context.Background()

	t.Run("unsupported type", func(t *testing.T) {
		called := false
		client := grail.NewClient(&mock.Provider{
			GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
				called = true
				return grail.Response{}, nil
			},
		})
		_, err := grail.JSON[schemaNode](ctx, client, "tree")
		if grail.GetErrorCode(err) != grail.InvalidArgument || !strings.Contains(err.Error(), "recursive type") {
			t.Fatalf("expected invalid_argument about recursion, got %v", err)
		}
		if called {
			t.Fatal("provider should not be called for an unsupported type")
		}
	})

	t.Run("provider error", func(t *testing.T) {
		providerErr := grail.NewGrailError(grail.RateLimited, "slow down").WithProviderName("mock")
		client := grail.NewClient(&mock.Provider{
			GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
				return grail.Response{}, providerErr
			},
		})
		if _, err := grail.JSON[recipe](ctx, client, "hi"); !errors.Is(err, providerErr) {
			t.Fatalf("expected provider error, got %v", err)
		}
	})

	t.Run("no JSON part", func(t *testing.T) {
		client := grail.NewClient(&mock.Provider{
			GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
				return grail.Response{
					Outputs: []grail.OutputPart{grail.NewTextOutputPart("not json")},
				}, nil
			},
		})
		_, err := grail.JSON[recipe](ctx, client, "hi")
		if grail.GetErrorCode(err) != grail.OutputInvalid {
			t.Fatalf("expected output_invalid, got %v", err)
		}
	})

	t.Run("wrong shape", func(t *testing.T) {
		client := grail.NewClient(&mock.Provider{
			GenerateFn: func(ctx context.Context, req grail.Request) (grail.Response, error) {
				return grail.Response{
					Outputs: []grail.OutputPart{grail.NewJSONOutputPart([]byte(`{"servings":"four"}`))},
				}, nil
			},
		})
		r, err := grail.JSON[recipe](ctx, client, "hi")
		if grail.GetErrorCode(err) != grail.OutputInvalid {
			t.Fatalf("expected output_invalid, got %v", err)
		}
		var typeErr *json.UnmarshalTypeError
		if !errors.As(err, &typeErr) {
			t.Fatalf("expected wrapped json.UnmarshalTypeError, got %v", err)
		}
		if !reflect.DeepEqual(r, recipe{}) {
			t.Fatalf("expected zero value on error, got %+v", r)
		}
	})
}

// assertJSONEqual compares got, marshaled to JSON, with the JSON text want.
func assertJSONEqual(t *testing.T, got any, want string) {
	t.Helper()
	gotBytes, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var g, w any
	if err := json.Unmarshal(gotBytes, &g); err != nil {
		t.Fatalf("unmarshal got: %v", err)
	}
	if err := json.Unmarshal([]byte(want), &w); err != nil {
		t.Fatalf("unmarshal want: %v", err)
	}
	if !reflect.DeepEqual(g, w) {
		t.Fatalf("schema mismatch\n got: %s\nwant: %s", gotBytes, want)
	}
}
