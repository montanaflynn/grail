package mock

import "github.com/montanaflynn/grail"

// Compile-time check that Provider implements grail.Provider.
var _ grail.Provider = (*Provider)(nil)
