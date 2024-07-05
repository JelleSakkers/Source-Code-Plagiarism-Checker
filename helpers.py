import re
import numpy as np
import html

from pygments import lex
from pygments.lexers import get_lexer_by_name
from pygments.token import Token, _TokenType


def get_token_coverage(k, token_len, indices):
    """
    Determine the amount of tokens in the filtered source code
    wich are included in the winnowed indices.
    """
    indices_arr = np.concatenate([np.array(i) for i in indices.values()])
    coverage = np.zeros(token_len)
    for offset in range(k):
        coverage[indices_arr + offset] = 1
    return np.sum(coverage)


def escape(input: str) -> str:
    """
    Helper function to escape raw code to HTML. Most work will be done by
    the escape function inside the 'html' module, except the escaping of
    newlines.

    Input:
        input: Some text or code that shall be HTML escaped.

    Output:
        HTML-safe escaped input text. Newlines are replaced with breaks as
        well.
    """
    output = html.escape(input)
    output = output.replace("\n", "<br>")
    return output


def simplify_code(source_code: str, language: str) -> tuple[str, np.ndarray]:
    """
    Function to reduce the given source code. It will replace functions,
    variables, classes and other types of tokens to just a simple character.
    Details like function or variables are lost.

    Input:
        source_code: The source code that should be reduced, it should be
                     given as a string.
        language   : Which programming language the source code is writen in.
                     Should be given as a string.

    Output:
        Simplified source code, it is given back as a string.
    """
    # Mapping from a token type to a simple character.
    TOKEN_TYPE_MAPPING = {
        Token.Name.Function: "f",
        Token.Name.Builtin: "f",
        Token.Name.Class: "o",
        Token.Name.Namespace: "n",
        Token.Name: "v",
        Token.Literal.String.Double: '"s"',
        Token.Literal.String: '"s"',
        Token.Comment.Preproc: "p",
        Token.Comment.PreprocFile: "i",
        Token.Literal.String.Char: "c",
    }
    lexer = get_lexer_by_name(language)
    tokens = lex(source_code, lexer)

    # A list to put every new simplified line of code in.
    simplified_code = []
    # Simple token type memory.
    prev_token_type = None

    # Helper function to reduce specific string type tokens to a simpler one.
    # Can also reduce magic functions to just a function token.
    def reduce_token(token_type: _TokenType):
        # Check if it is a string before we reduce one string type to another.
        if token_type in Token.Literal.String:
            # Change single quote to double.
            token_type = (
                Token.Literal.String.Double
                if token_type == Token.Literal.String.Single
                else token_type
            )

            # Change escaped token type to string literal if it is inside the
            # quotes.
            if (
                prev_token_type is not None
                and (
                    prev_token_type == Token.Literal.String.Double
                    or prev_token_type == Token.Literal.String
                )
                and token_type == Token.Literal.String.Escape
            ):
                token_type = prev_token_type

            return token_type
        # Reduce magic functions to just functions.
        elif token_type in Token.Name.Function:
            return Token.Name.Function
        return token_type

    # To make a mapping from the transformed text possible to the original text
    # We need to remember our position in both the original and transformed
    # text. We should not remember the positions of whitespace characters (
    # including newlines and other similair control characters).
    orig_file_offset = 0
    trans_file_offset = 0
    # Token map interface: [[<start offset original token>,
    #                        <end offset original token>,
    #                        <start offset transformed token>,
    #                        <end offset transformed token>]
    #                       , ...]
    # The end offset should be included so printing an string slice should be
    # done like this:
    # print(source_code
    #       [<start offset original token>:<end offset original token> + 1])
    token_mapping = []

    # Helper function to update our file mapping. Will skip whitespace and
    # similair characters.
    def update_map(token_value: str,
                   token_len: int,
                   trans_token_len: int) -> None:
        # Skip whitespaces.
        if re.match(r"\s+", token_value) is None:
            token_mapping.append(
                [
                    orig_file_offset,
                    orig_file_offset + token_len - 1,
                    trans_file_offset,
                    trans_file_offset + trans_token_len - 1,
                ]
            )

    for token_type, token_value in tokens:
        token_len = len(token_value)

        if (token_type := reduce_token(token_type)) in TOKEN_TYPE_MAPPING and \
                not (token_type == Token.Comment.Preproc and
                     token_value == "\n"):
            # Skip if we have already seen this token type the previous round,
            # otherwise we get ugly 'sss' strings for example.
            if token_type == prev_token_type:
                # Update the original end of the most recent mapped token.
                token_mapping[-1][1] += token_len
                # Don't forget to update the file offset.
                orig_file_offset += token_len
                continue
            trans_token = TOKEN_TYPE_MAPPING[token_type]
            trans_token_len = len(trans_token)
            simplified_code.append(trans_token)
            # Remember previous token type.
            prev_token_type = token_type
            # Update our content mapping.
            update_map(token_value, token_len, trans_token_len)
            # Most but not all replacement tokens are just one character long.
            trans_file_offset += trans_token_len
        # Skip comments (except preprocessor stuff), since these are not
        # that important for the plagiarism check.
        elif (
            token_type in Token.Comment
            and token_type != Token.Comment.Preproc
            and token_type != Token.Comment.PreprocFile
        ):
            pass
        else:
            # Make sure to reset if an unused token type comes by.
            prev_token_type = None
            simplified_code.append(token_value)
            # Update our content mapping.
            update_map(token_value, token_len, token_len)
            # Update transformed file mapping with the original token length if
            # it parsed as is.
            trans_file_offset += token_len
        # Always update the original file offset.
        orig_file_offset += token_len

    # Create a string out of a list. Return both the string and the content
    # mapper.
    return "".join(simplified_code), np.array(token_mapping)
