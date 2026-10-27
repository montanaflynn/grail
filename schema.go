package grail

import (
	"encoding/json"
	"fmt"
)

// JSONSchemaMap normalizes an OutputJSON schema into a JSON Schema object for
// provider structured-output APIs. A map[string]any is returned as is; any
// other value is marshaled and decoded as a JSON object. A nil schema, or one
// that marshals to null, returns nil. A schema that is not a JSON object
// returns an error.
func JSONSchemaMap(schema any) (map[string]any, error) {
	if schema == nil {
		return nil, nil
	}
	if m, ok := schema.(map[string]any); ok {
		return m, nil
	}
	b, err := json.Marshal(schema)
	if err != nil {
		return nil, err
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("schema must be a JSON object: %w", err)
	}
	return m, nil
}
