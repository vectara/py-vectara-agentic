"""Vectara Hallucination Detection and Correction client."""

import logging
import os
from typing import List, Dict, Optional, Tuple

import requests
from llama_index.core.llms import MessageRole

def convert_table_to_sentences(table_text: str) -> str:
    """
    Convert a markdown table to natural language sentences.

    Args:
        table_text: Raw markdown table text with pipes and separators

    Returns:
        str: Natural language sentences describing the table data
    """
    import re

    lines = table_text.strip().split('\n')

    # Skip separator lines (lines with |---|---|)
    content_lines = [line for line in lines if not re.match(r'^\s*\|[\s\-\|:]+\|\s*$', line)]

    if len(content_lines) < 2:
        # Not enough content for header + data
        return table_text

    # Parse table structure
    parsed_rows = []
    for line in content_lines:
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|') if cell.strip()]
            if cells:
                parsed_rows.append(cells)

    if len(parsed_rows) < 2:
        return table_text

    headers = parsed_rows[0]
    data_rows = parsed_rows[1:]

    # Generate sentences for each data row
    sentences = []

    for row in data_rows:
        if len(row) != len(headers):
            # Handle mismatched columns by padding or truncating
            while len(row) < len(headers):
                row.append("N/A")
            row = row[:len(headers)]

        # Create contextual sentences based on common patterns
        sentence = create_row_sentence(headers, row)
        if sentence:
            sentences.append(sentence)

    return ' '.join(sentences) if sentences else table_text


def create_row_sentence(headers: list, row: list) -> str:
    """
    Create a natural language sentence from table headers and row data.

    Args:
        headers: List of column headers
        row: List of row values

    Returns:
        str: Natural language sentence describing the row
    """
    if not headers or not row:
        return ""

    # Create data dictionary for easier access
    data = {headers[i].lower(): row[i] for i in range(min(len(headers), len(row)))}

    # Detect common table patterns and generate appropriate sentences

    # Vehicle comparison tables (common in automotive data)
    if any(key in data for key in ['vehicle', 'vehicle type', 'type']):
        return create_vehicle_sentence(headers, row, data)

    # Financial/pricing tables
    elif any(key in data for key in ['price', 'cost', 'amount', 'value']):
        return create_financial_sentence(headers, row, data)

    # Performance/specification tables
    elif any(key in data for key in ['range', 'speed', 'power', 'capacity']):
        return create_performance_sentence(headers, row, data)

    # Generic fallback - create descriptive sentence
    else:
        return create_generic_sentence(headers, row)


def create_vehicle_sentence(headers: list, row: list, data: dict) -> str:
    """Create sentence for vehicle comparison data."""
    # Get vehicle identifier
    vehicle = (data.get('vehicle type') or data.get('vehicle') or
              data.get('type') or row[0])

    parts = [f"{vehicle} vehicles"]

    # Add range information
    if 'range' in data or 'range (miles)' in data:
        range_val = data.get('range (miles)') or data.get('range')
        if range_val and range_val.lower() not in ['n/a', 'na', '-', '']:
            if 'electric' in range_val:
                parts.append(f"have an electric range of {range_val}")
            else:
                parts.append(f"have a range of {range_val} miles")
        else:
            parts.append("do not have electric-only range")

    # Add price information
    price_keys = ['price range', 'price', 'cost', 'cost range']
    for key in price_keys:
        if key in data:
            price_val = data[key]
            if price_val and price_val.lower() not in ['n/a', 'na', '-', '']:
                parts.append(f"and cost {price_val}")
            break

    # Add other specifications
    for i, header in enumerate(headers):
        header_lower = header.lower()
        if (header_lower not in ['vehicle type', 'vehicle', 'type', 'range', 'range (miles)', 'price range', 'price', 'cost'] and
            i < len(row) and row[i] and row[i].lower() not in ['n/a', 'na', '-', '']):
            parts.append(f"with {header.lower()} of {row[i]}")

    return ' '.join(parts) + '.'


def create_financial_sentence(headers: list, row: list, data: dict) -> str:
    """Create sentence for financial/pricing data."""
    # Get primary identifier (usually first column)
    identifier = row[0]

    parts = [f"{identifier}"]

    # Add price/cost information
    price_keys = ['price', 'cost', 'amount', 'value', 'price range', 'cost range']
    for key in price_keys:
        if key in data:
            price_val = data[key]
            if price_val and price_val.lower() not in ['n/a', 'na', '-', '']:
                if 'range' in key:
                    parts.append(f"costs {price_val}")
                else:
                    parts.append(f"has a {key} of {price_val}")
            break

    # Add other financial attributes
    for i, header in enumerate(headers[1:], 1):  # Skip first column (identifier)
        header_lower = header.lower()
        if (header_lower not in price_keys and i < len(row) and
            row[i] and row[i].lower() not in ['n/a', 'na', '-', '']):
            parts.append(f"and {header.lower()} of {row[i]}")

    return ' '.join(parts) + '.'


def create_performance_sentence(headers: list, row: list, data: dict) -> str:
    """Create sentence for performance/specification data."""
    identifier = row[0]

    parts = [f"{identifier}"]

    # Add performance metrics
    perf_attributes = []
    for i, header in enumerate(headers[1:], 1):  # Skip identifier
        if i < len(row) and row[i] and row[i].lower() not in ['n/a', 'na', '-', '']:
            perf_attributes.append(f"{header.lower()} of {row[i]}")

    if perf_attributes:
        if len(perf_attributes) == 1:
            parts.append(f"has {perf_attributes[0]}")
        else:
            parts.append(f"has {', '.join(perf_attributes[:-1])} and {perf_attributes[-1]}")

    return ' '.join(parts) + '.'


def create_generic_sentence(headers: list, row: list) -> str:
    """Create generic descriptive sentence for any table structure."""
    if not row:
        return ""

    identifier = row[0]
    parts = [f"{identifier}"]

    # Add attributes from remaining columns
    attributes = []
    for i, header in enumerate(headers[1:], 1):
        if i < len(row) and row[i] and row[i].lower() not in ['n/a', 'na', '-', '']:
            # Improve grammar by using "of" instead of "is" for most attributes
            attr_value = row[i]
            header_lower = header.lower()

            # Use more natural phrasing
            if any(word in header_lower for word in ['speed', 'size', 'capacity', 'memory', 'storage']):
                attributes.append(f"{header_lower} of {attr_value}")
            elif any(word in header_lower for word in ['level', 'type', 'status', 'category']):
                attributes.append(f"{header_lower} of {attr_value}")
            else:
                attributes.append(f"{header_lower} of {attr_value}")

    if attributes:
        if len(attributes) == 1:
            parts.append(f"has {attributes[0]}")
        else:
            parts.append(f"has {', '.join(attributes[:-1])} and {attributes[-1]}")

    return ' '.join(parts) + '.'


def remove_citations(text: str) -> str:
    """
    Remove citations from text in both markdown link and numeric reference formats.

    Args:
        text: Input text potentially containing citations

    Returns:
        str: Text with citations removed
    """
    import re

    # Remove markdown-style citations: [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove all remaining square bracket citations: [text], [1], etc.
    text = re.sub(r'\[[^\]]+\]', '', text)

    # Clean up extra whitespace that may result from removal
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def markdown_to_text(md: str) -> str:
    """
    Convert a Markdown-formatted string into plain text using HTML-first approach.

    This replaces the CommonMark-based approach that had issues with mixed content.
    Uses a two-step process: Markdown → HTML → Clean Text for better reliability.
    """
    import re

    # Check if the input is purely a table (common case for FCS)
    table_pattern = r'^\s*(\|[^\n]*\|(?:\n\|[^\n]*\|)*)\s*$'
    table_match = re.match(table_pattern, md.strip(), flags=re.MULTILINE | re.DOTALL)

    if table_match:
        # Pure table - convert directly to natural language
        table_text = table_match.group(1)
        return convert_table_to_sentences(table_text)

    # Mixed content - use HTML-first approach
    # Step 1: Convert tables to natural language sentences first
    def convert_table(match):
        table_text = match.group(0)
        return convert_table_to_sentences(table_text)

    mixed_table_pattern = r'(\|[^\n]*\|(?:\n\|[^\n]*\|)+)'
    md_with_tables_converted = re.sub(mixed_table_pattern, convert_table, md, flags=re.MULTILINE)

    # Step 2: Use HTML-first approach for reliable mixed content processing
    try:
        # Try mistune first (fastest and most reliable)
        import mistune
        from bs4 import BeautifulSoup

        # Markdown → HTML (robust parsing)
        html = mistune.html(md_with_tables_converted)

        # HTML → Plain Text (using BeautifulSoup for proper text extraction)
        soup = BeautifulSoup(html, 'html.parser')
        clean_text = soup.get_text()

        # Clean up the output
        lines = [line.strip() for line in clean_text.split('\n')]
        # Remove excessive empty lines but preserve paragraph breaks
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')
                prev_empty = True

        return '\n'.join(cleaned_lines).strip()

    except ImportError:
        # Fallback: Try standard markdown library with BeautifulSoup
        try:
            import markdown
            from bs4 import BeautifulSoup

            md_parser = markdown.Markdown()
            html = md_parser.convert(md_with_tables_converted)

            soup = BeautifulSoup(html, 'html.parser')
            clean_text = soup.get_text()

            # Clean up the output
            lines = [line.strip() for line in clean_text.split('\n')]
            cleaned_lines = []
            prev_empty = False
            for line in lines:
                if line:
                    cleaned_lines.append(line)
                    prev_empty = False
                elif not prev_empty:
                    cleaned_lines.append('')
                    prev_empty = True

            return '\n'.join(cleaned_lines).strip()

        except ImportError:
            # Last resort: Simple regex cleanup (basic but functional)
            logging.warning("Neither mistune nor markdown libraries available, using basic text processing")

            # Remove markdown formatting
            result = md_with_tables_converted
            result = re.sub(r'^#+\s*(.+)$', r'\1', result, flags=re.MULTILINE)  # Headers
            result = re.sub(r'\*\*(.+?)\*\*', r'\1', result)  # Bold
            result = re.sub(r'\*(.+?)\*', r'\1', result)  # Italic
            result = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', result)  # Links
            result = re.sub(r'`([^`]+)`', r'\1', result)  # Inline code

            # Clean up whitespace
            lines = [line.strip() for line in result.split('\n')]
            cleaned_lines = []
            prev_empty = False
            for line in lines:
                if line:
                    cleaned_lines.append(line)
                    prev_empty = False
                elif not prev_empty:
                    cleaned_lines.append('')
                    prev_empty = True

            return '\n'.join(cleaned_lines).strip()


class Hallucination:
    """Vectara Hallucination Detection and Correction client."""

    def __init__(self, vectara_api_key: str):
        self._vectara_api_key = vectara_api_key

    def compute(self, context: list[str], hypothesis: str) -> float:
        """
        Calls the Vectara Hallucination Detection endpoint to evaluate the factual consistency of a hypothesis against a given context.

        Parameters:
            context (list[str]): The source texs against which the hypothesis will be evaluated.
            hypothesis (str): The generated text to be evaluated for factual consistency.

        Returns:
            float: The factual consistency score rounded to four decimal places.

        Raises:
            requests.exceptions.RequestException: If there is a network-related error or the API call fails.
        """

        # clean response from any markdown or other formatting, then remove citations
        try:
            clean_hypothesis = remove_citations(markdown_to_text(hypothesis))
            clean_context = [remove_citations(markdown_to_text(text)) for text in context]
        except Exception as e:
            raise ValueError(f"Markdown parsing of hypothesis failed: {e}") from e

        logging.info(f"🔍 [HALLUCINATION_DEBUG] Cleaned hypothesis: {clean_hypothesis}\n\n")
        logging.info("🔍 [HALLUCINATION_DEBUG] Context:")
        for i,ctx in enumerate(clean_context):
            logging.info(f"🔍 [HALLUCINATION_DEBUG] context {i}: {ctx}\n\n")

        # compute hallucination score with Vectara endpoint
        payload = {
            "model_parameters": {"model_name": "hhem_v2.3"},
            "generated_text": clean_hypothesis,
            "source_texts": clean_context,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "x-api-key": self._vectara_api_key,
        }

        response = requests.post(
            "https://api.vectara.io/v2/evaluate_factual_consistency",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return round(data.get("score", 0.0), 4)

    def compute_with_vhc(self, query: str, context: list[str], hypothesis: str) -> Tuple[str, list[str]]:
        """
        Calls the Vectara VHC (Vectara Hallucination Correction)

        Returns:
            float: The factual consistency score rounded to four decimal places.
            str: The corrected hypothesis text.
        """

        payload = {
            "generated_text": hypothesis,
            "query": query,
            "documents": [{"text": c} for c in context],
            "model_name": "vhc-large-1.0"
        }
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'x-api-key': os.getenv("VECTARA_API_KEY")
        }

        response = requests.post(
            "https://api.vectara.io/v2/hallucination_correctors/correct_hallucination",
            json=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        corrected_text = data.get("corrected_text", "")
        corrections = data.get("corrections", [])
        return corrected_text, corrections


def extract_tool_call_mapping(chat_history) -> Dict[str, str]:
    """Extract tool_call_id to tool_name mapping from chat history."""
    tool_call_id_to_name = {}
    for msg in chat_history:
        if msg.role == MessageRole.ASSISTANT and hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
            tool_calls = msg.additional_kwargs.get('tool_calls', [])
            for tool_call in tool_calls:
                if isinstance(tool_call, dict) and 'id' in tool_call and 'function' in tool_call:
                    tool_call_id = tool_call['id']
                    tool_name = tool_call['function'].get('name')
                    if tool_call_id and tool_name:
                        tool_call_id_to_name[tool_call_id] = tool_name

    return tool_call_id_to_name


def identify_tool_name(msg, tool_call_id_to_name: Dict[str, str]) -> Optional[str]:
    """Identify tool name from message using multiple strategies."""
    tool_name = None

    # First try: standard tool_name attribute (for backwards compatibility)
    tool_name = getattr(msg, 'tool_name', None)

    # Second try: additional_kwargs (LlamaIndex standard location)
    if tool_name is None and hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
        tool_name = msg.additional_kwargs.get('name') or msg.additional_kwargs.get('tool_name')

        # If no direct tool name, try to map from tool_call_id
        if tool_name is None:
            tool_call_id = msg.additional_kwargs.get('tool_call_id')
            if tool_call_id and tool_call_id in tool_call_id_to_name:
                tool_name = tool_call_id_to_name[tool_call_id]

    # Third try: extract from content if it's a ToolOutput object
    if tool_name is None and hasattr(msg.content, 'tool_name'):
        tool_name = msg.content.tool_name

    return tool_name


def check_tool_fcs_eligibility(tool_name: Optional[str], tools: List) -> bool:
    """Check if a tool is FCS-eligible by looking up in tools list."""
    if not tool_name or not tools:
        return True

    # Try to find the tool and check its FCS eligibility
    for tool in tools:
        if hasattr(tool, 'metadata') and hasattr(tool.metadata, 'name') and tool.metadata.name == tool_name:
            if hasattr(tool.metadata, 'fcs_eligible'):
                is_fcs_eligible = tool.metadata.fcs_eligible
                return is_fcs_eligible
            break

    return True


def analyze_hallucinations(
    chat_history,
    agent_response: str,
    tools: List,
    vectara_api_key: str
) -> Tuple[float, str, List[str]]:
    """Calculate FCS score from chat history and agent response."""
    if not vectara_api_key:
        logging.debug("No Vectara API key - returning None")
        return None

    # Build a mapping from tool_call_id to tool_name for better tool identification
    tool_call_id_to_name = extract_tool_call_mapping(chat_history)

    context = []
    last_assistant_index = -1
    for i, msg in enumerate(chat_history):
        if msg.role == MessageRole.ASSISTANT and msg.content:
            last_assistant_index = i

    for i, msg in enumerate(chat_history):
        if msg.role == MessageRole.TOOL:
            # Check if this tool call should be included in FCS calculation
            tool_name = identify_tool_name(msg, tool_call_id_to_name)
            is_fcs_eligible = check_tool_fcs_eligibility(tool_name, tools)

            # Only count tool calls from FCS-eligible tools
            if is_fcs_eligible:
                content = msg.content

                # Since tools with human-readable output now convert to formatted strings immediately
                # in VectaraTool._format_tool_output(), we just use the content directly
                content = str(content) if content is not None else ""

                # Only add non-empty content to context
                if content and content.strip():
                    context.append(content)

        elif msg.role == MessageRole.USER and msg.content:
            context.append(msg.content)

        elif msg.role == MessageRole.ASSISTANT and msg.content:
            if i == last_assistant_index:   # do not include the last assistant message
                continue
            context.append(msg.content)

    # If no context or no tool calls, we cannot compute FCS
    if len(context)==0:
        return None, None, []

    try:
        score = Hallucination(vectara_api_key).compute(context, agent_response)
        corrected_text, corrections = Hallucination(vectara_api_key).compute_with_vhc(
            query=agent_response,
            context=context,
            hypothesis=agent_response
        )
        return score, corrected_text, corrections

    except Exception as e:
        logging.error(
            f"Hallucination computation failed: {e}. "
            "Ensure you have a valid Vectara API key and the Hallucination Detection service is available."
        )
        return None
