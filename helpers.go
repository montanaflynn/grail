package grail

import "context"

// Text generates text from prompt and returns it. It is a thin wrapper over
// Generate with OutputText. It returns an OutputInvalid error if the response
// has no text part.
//
//	text, err := client.Text(ctx, "Say hello", grail.WithTier(grail.Fast))
func (c *Client) Text(ctx context.Context, prompt string, opts ...CallOption) (string, error) {
	res, err := c.Generate(ctx, buildRequest(prompt, OutputText(), opts))
	if err != nil {
		return "", err
	}
	text, ok := res.Text()
	if !ok {
		return "", NewGrailError(OutputInvalid, "no text output part found in response").WithProviderName(res.Provider.Name)
	}
	return text, nil
}

// Image generates a single image from prompt and returns its bytes. It is a
// thin wrapper over Generate with OutputImage. It returns an OutputInvalid
// error if the response has no image part.
//
//	png, err := client.Image(ctx, "A red fox", grail.WithTier(grail.Best))
func (c *Client) Image(ctx context.Context, prompt string, opts ...CallOption) ([]byte, error) {
	res, err := c.Generate(ctx, buildRequest(prompt, OutputImage(ImageSpec{Count: 1}), opts))
	if err != nil {
		return nil, err
	}
	images, ok := res.Images()
	if !ok {
		return nil, NewGrailError(OutputInvalid, "no image output part found in response").WithProviderName(res.Provider.Name)
	}
	return images[0], nil
}
