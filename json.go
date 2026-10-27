package grail

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// JSON generates structured output shaped like T and decodes it. The JSON
// Schema is derived from T with SchemaFor and sent with OutputJSON, so
// providers with native structured output constrain the response to it.
//
// It returns an InvalidArgument error if no schema can be derived from T (see
// SchemaFor), and an OutputInvalid error if the response has no JSON part or
// the JSON does not decode into T.
//
//	type Recipe struct {
//		Name        string   `json:"name"`
//		Ingredients []string `json:"ingredients"`
//	}
//	recipe, err := grail.JSON[Recipe](ctx, client, "A pancake recipe")
func JSON[T any](ctx context.Context, c *Client, prompt string, opts ...CallOption) (T, error) {
	var zero T
	schema, err := schemaForType(reflect.TypeFor[T]())
	if err != nil {
		return zero, NewGrailError(InvalidArgument, fmt.Sprintf("derive JSON schema: %v", err)).WithCause(err)
	}

	res, err := c.Generate(ctx, buildRequest(prompt, OutputJSON(schema), opts))
	if err != nil {
		return zero, err
	}

	var out T
	if err := res.DecodeJSON(&out); err != nil {
		var ge GrailError
		if errors.As(err, &ge) { // no JSON part
			return zero, NewGrailError(OutputInvalid, "no JSON output part found in response").WithProviderName(res.Provider.Name)
		}
		return zero, NewGrailError(OutputInvalid, fmt.Sprintf("decode JSON output into %s: %v", reflect.TypeFor[T](), err)).
			WithCause(err).WithProviderName(res.Provider.Name)
	}
	return out, nil
}

// SchemaFor derives a JSON Schema object from the Go type T by reflection, for
// use with OutputJSON. It follows encoding/json's view of the type:
//
//   - Structs become objects with additionalProperties false. Field names come
//     from the json tag, "-" fields and unexported fields are skipped, and
//     embedded structs have their fields promoted.
//   - Slices and arrays become arrays ([]byte is a base64 string), maps with
//     string keys become objects whose additionalProperties is the value
//     schema, and time.Time is a string with format date-time.
//   - Booleans, integers, floats and strings map to their JSON types. A
//     ",string" json option makes a scalar field a string.
//
// Every property is listed in required, as OpenAI strict mode demands.
// Optional fields (pointers, and fields tagged omitempty or omitzero) are
// instead nullable: their type becomes e.g. ["string", "null"]. A null decodes
// to the field's zero value, so the result is the same as an absent field.
//
// A jsonschema struct tag adds a description and an enum (values separated by
// "|", applied to the elements of a slice field):
//
//	Status string `json:"status" jsonschema:"description=Current state,enum=open|closed"`
//
// Maps produce schemas that OpenAI strict mode rejects, since it only allows
// additionalProperties false; use structs or slices of key/value structs there.
//
// SchemaFor panics if T cannot be described: recursive types, interfaces,
// channels, functions, complex numbers, and maps without string keys. JSON
// returns an error instead.
func SchemaFor[T any]() map[string]any {
	schema, err := schemaForType(reflect.TypeFor[T]())
	if err != nil {
		panic("grail: SchemaFor: " + err.Error())
	}
	return schema
}

var timeType = reflect.TypeFor[time.Time]()

// schemaForType derives the schema for a top-level type. A pointer at the
// root is dereferenced instead of made nullable, since providers require the
// root to be a concrete type.
func schemaForType(t reflect.Type) (map[string]any, error) {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	b := &schemaBuilder{visiting: map[reflect.Type]bool{}}
	return b.schema(t)
}

// schemaBuilder tracks the struct types on the current path to detect
// recursion. Repeated, non-recursive use of a type is inlined each time.
type schemaBuilder struct {
	visiting map[reflect.Type]bool
}

func (b *schemaBuilder) schema(t reflect.Type) (map[string]any, error) {
	if t.Kind() == reflect.Pointer {
		for t.Kind() == reflect.Pointer {
			t = t.Elem()
		}
		s, err := b.schema(t)
		if err != nil {
			return nil, err
		}
		return nullable(s), nil
	}
	if t == timeType {
		return map[string]any{"type": "string", "format": "date-time"}, nil
	}

	switch t.Kind() {
	case reflect.Bool:
		return map[string]any{"type": "boolean"}, nil
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return map[string]any{"type": "integer"}, nil
	case reflect.Float32, reflect.Float64:
		return map[string]any{"type": "number"}, nil
	case reflect.String:
		return map[string]any{"type": "string"}, nil
	case reflect.Slice, reflect.Array:
		if t.Kind() == reflect.Slice && t.Elem().Kind() == reflect.Uint8 {
			return map[string]any{"type": "string"}, nil
		}
		items, err := b.schema(t.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "array", "items": items}, nil
	case reflect.Map:
		if t.Key().Kind() != reflect.String {
			return nil, fmt.Errorf("map key type %s is not supported, only string keys", t.Key())
		}
		values, err := b.schema(t.Elem())
		if err != nil {
			return nil, err
		}
		return map[string]any{"type": "object", "additionalProperties": values}, nil
	case reflect.Struct:
		return b.structSchema(t)
	default:
		return nil, fmt.Errorf("type %s (kind %s) is not supported", t, t.Kind())
	}
}

func (b *schemaBuilder) structSchema(t reflect.Type) (map[string]any, error) {
	fields, err := b.collectFields(t, 0, false)
	if err != nil {
		return nil, err
	}

	// Like encoding/json, a shallower field hides promoted fields with the
	// same name. Among fields at the same depth, the first one wins.
	depth := map[string]int{}
	for _, f := range fields {
		if d, ok := depth[f.name]; !ok || f.depth < d {
			depth[f.name] = f.depth
		}
	}

	properties := map[string]any{}
	required := []string{}
	for _, f := range fields {
		if _, seen := properties[f.name]; seen || f.depth != depth[f.name] {
			continue
		}
		properties[f.name] = f.schema
		required = append(required, f.name)
	}

	return map[string]any{
		"type":                 "object",
		"properties":           properties,
		"required":             required,
		"additionalProperties": false,
	}, nil
}

type schemaField struct {
	name   string
	depth  int
	schema map[string]any
}

// collectFields returns the JSON fields of struct type t in declaration
// order, with embedded struct fields promoted at depth+1. Fields reached
// through an embedded pointer are optional, since a nil pointer omits them.
func (b *schemaBuilder) collectFields(t reflect.Type, depth int, optional bool) ([]schemaField, error) {
	if b.visiting[t] {
		return nil, fmt.Errorf("recursive type %s is not supported", t)
	}
	b.visiting[t] = true
	defer delete(b.visiting, t)

	var fields []schemaField
	for i := range t.NumField() {
		f := t.Field(i)
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		name, opts, _ := strings.Cut(tag, ",")

		ft := f.Type
		isPtr := ft.Kind() == reflect.Pointer
		for ft.Kind() == reflect.Pointer {
			ft = ft.Elem()
		}

		if f.Anonymous && name == "" && ft.Kind() == reflect.Struct {
			// encoding/json ignores embedded unexported pointers to structs.
			if !f.IsExported() && isPtr {
				continue
			}
			promoted, err := b.collectFields(ft, depth+1, optional || isPtr)
			if err != nil {
				return nil, err
			}
			fields = append(fields, promoted...)
			continue
		}
		if !f.IsExported() {
			continue
		}
		if name == "" {
			name = f.Name
		}

		s, err := b.schema(ft)
		if err != nil {
			return nil, fmt.Errorf("field %s.%s: %w", t, f.Name, err)
		}
		if hasOption(opts, "string") && isScalar(ft.Kind()) {
			s = map[string]any{"type": "string"}
		}
		if err := applySchemaTag(s, f.Tag.Get("jsonschema")); err != nil {
			return nil, fmt.Errorf("field %s.%s: %w", t, f.Name, err)
		}
		if optional || isPtr || hasOption(opts, "omitempty") || hasOption(opts, "omitzero") {
			s = nullable(s)
		}
		fields = append(fields, schemaField{name: name, depth: depth, schema: s})
	}
	return fields, nil
}

// applySchemaTag applies a jsonschema struct tag of comma-separated key=value
// pairs to s. A segment without a known key continues the previous value, so
// descriptions may contain commas.
func applySchemaTag(s map[string]any, tag string) error {
	if tag == "" {
		return nil
	}
	values := map[string]string{}
	var key string
	for seg := range strings.SplitSeq(tag, ",") {
		k, v, ok := strings.Cut(seg, "=")
		if ok && (k == "description" || k == "enum") {
			key = k
			values[key] = v
			continue
		}
		if key == "" {
			return fmt.Errorf("unknown jsonschema tag key in %q", seg)
		}
		values[key] += "," + seg
	}

	if desc, ok := values["description"]; ok {
		s["description"] = desc
	}
	if enum, ok := values["enum"]; ok {
		target := s
		if items, ok := s["items"].(map[string]any); ok && s["type"] == "array" {
			target = items
		}
		vals, err := enumValues(target["type"], strings.Split(enum, "|"))
		if err != nil {
			return err
		}
		target["enum"] = vals
	}
	return nil
}

// enumValues converts enum tag values to the JSON type of the schema they
// constrain.
func enumValues(typ any, raw []string) ([]any, error) {
	vals := make([]any, 0, len(raw))
	for _, r := range raw {
		var (
			v   any
			err error
		)
		switch typ {
		case "string":
			v = r
		case "integer":
			v, err = strconv.ParseInt(r, 10, 64)
		case "number":
			v, err = strconv.ParseFloat(r, 64)
		case "boolean":
			v, err = strconv.ParseBool(r)
		default:
			return nil, fmt.Errorf("enum is not supported on type %v", typ)
		}
		if err != nil {
			return nil, fmt.Errorf("enum value %q is not a valid %v", r, typ)
		}
		vals = append(vals, v)
	}
	return vals, nil
}

// nullable lets s also match null by adding "null" to its type, and to its
// enum when it has one.
func nullable(s map[string]any) map[string]any {
	typ, ok := s["type"].(string)
	if !ok {
		return s // already nullable
	}
	s["type"] = []any{typ, "null"}
	if enum, ok := s["enum"].([]any); ok {
		s["enum"] = append(enum, nil)
	}
	return s
}

func hasOption(opts, name string) bool {
	for opt := range strings.SplitSeq(opts, ",") {
		if opt == name {
			return true
		}
	}
	return false
}

func isScalar(k reflect.Kind) bool {
	switch k {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr,
		reflect.Float32, reflect.Float64, reflect.String:
		return true
	}
	return false
}
