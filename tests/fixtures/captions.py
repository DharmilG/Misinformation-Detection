"""Caption corpora for the encoder tests.

The long caption deliberately exceeds CLIP's 77-token context window, so the
truncation path is exercised.  The corpus itself never does - document 02 §8.2
records that the corpus captions top out at 30 words - so this is the only
place that path is reached.
"""

SHORT = "A crowd gathers outside a courthouse on a rainy afternoon."

UNICODE = "Des Nd supposeunyana okusincedwa phosengula eMva\n\U0001f468‍\U0001f469‍\U0001f467"

#: Comfortably longer than 77 CLIP tokens.
LONG = (
    "A very long caption that is used to exercise the truncation path of the "
    "text encoder, because the context window of the model is limited to "
    "seventy seven tokens and anything beyond that has to be truncated rather "
    "than rejected, and the remaining tokens must not change the position at "
    "which the end of text token is pooled from, which is why the padding "
    "strategy is fixed at the maximum length of the window rather than at the "
    "length of the longest item in the batch, so that the result of encoding a "
    "single long caption does not depend on what else happens to be encoded "
    "beside it in the same batch, which is the property that the alignment "
    "guarantee in the encoder contract depends upon."
)

EMPTY = ""

WHITESPACE = "   \t  \n "


def approx_tokens(text, chars_per_token=4):
    """A rough token estimate, used to build captions that exceed the window."""
    return max(1, len(text) // chars_per_token)
