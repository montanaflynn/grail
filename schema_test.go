package grail

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestJSONSchemaMap(t *testing.T) {
	type schemaStruct struct {
		Type     string   `json:"type"`
		Required []string `json:"required"`
	}
	want := map[string]any{"type": "object", "required": []any{"name"}}

	tests := []struct {
		name    string
		schema  any
		want    map[string]any
		wantErr bool
	}{
		{"nil", nil, nil, false},
		{"map", want, want, false},
		{"struct", schemaStruct{Type: "object", Required: []string{"name"}}, want, false},
		{"raw message", json.RawMessage(`{"type":"object","required":["name"]}`), want, false},
		{"marshals to null", (*schemaStruct)(nil), nil, false},
		{"empty raw message", json.RawMessage(nil), nil, false},
		{"string", "not a schema", nil, true},
		{"array", []any{"object"}, nil, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := JSONSchemaMap(tt.schema)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("got %v, want %v", got, tt.want)
			}
		})
	}
}
