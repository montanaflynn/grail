package grail

import (
	"testing"
	"time"
)

func TestModel_IsDeprecated(t *testing.T) {
	if (Model{Name: "current"}).IsDeprecated() {
		t.Fatalf("model without a deprecation record should not be deprecated")
	}
	if !(Model{Name: "old", Deprecation: &Deprecation{Replacement: "new"}}).IsDeprecated() {
		t.Fatalf("model with a deprecation record should be deprecated")
	}
}

func TestModel_IsShutDown(t *testing.T) {
	shutdown := time.Date(2026, time.October, 23, 0, 0, 0, 0, time.UTC)
	dated := Model{Name: "old", Deprecation: &Deprecation{Shutdown: shutdown, Replacement: "new"}}
	undated := Model{Name: "old", Deprecation: &Deprecation{Replacement: "new"}}
	current := Model{Name: "current"}

	tests := []struct {
		name  string
		model Model
		now   time.Time
		want  bool
	}{
		{"not deprecated", current, shutdown.AddDate(10, 0, 0), false},
		{"no shutdown date", undated, shutdown.AddDate(10, 0, 0), false},
		{"before shutdown", dated, shutdown.Add(-time.Second), false},
		{"at shutdown", dated, shutdown, true},
		{"after shutdown", dated, shutdown.Add(time.Hour), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.model.IsShutDown(tt.now); got != tt.want {
				t.Fatalf("IsShutDown(%v) = %v, want %v", tt.now, got, tt.want)
			}
		})
	}
}
