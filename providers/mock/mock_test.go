package mock

import "github.com/montanaflynn/grail"

// Compile-time check that Provider implements grail.Provider.
var _ grail.Provider = (*Provider)(nil)

// Compile-time checks that CatalogProvider can be routed by role and tier.
var (
	_ grail.ProviderExecutor = (*CatalogProvider)(nil)
	_ grail.ModelResolver    = (*CatalogProvider)(nil)
	_ grail.ModelLister      = (*CatalogProvider)(nil)
)

// Compile-time check that StreamProvider can stream.
var _ grail.StreamExecutor = (*StreamProvider)(nil)
