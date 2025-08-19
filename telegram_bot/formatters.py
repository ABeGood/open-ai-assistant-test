"""
Telegram message formatting utilities.
"""

from typing import List, Tuple


def format_telegram_message(response_data) -> Tuple[str, List[str]]:
    """
    Format response data for Telegram API with markdown-compatible formatting.
    
    Args:
        response_data: SpecialistResponse or CombinatorResponse object
        
    Returns:
        Tuple[str, List[str]]: (formatted_markdown_text, images_list)
    """
    
    # Extract main response text
    response_text = response_data.response or ''
    sources = response_data.sources or []
    images = response_data.images or []
    
    # Clean and format the response text for Telegram markdown
    formatted_response = escape_telegram_markdown(response_text)
    
    # Build the complete message
    message_parts = []
    
    # Add the main response
    if formatted_response:
        message_parts.append(formatted_response)
    
    # Add sources section if sources exist
    if sources:
        message_parts.append("\n\n*Источники:*")
        
        # Format each source as a markdown link on a new line
        for i, source in enumerate(sources, 1):
            if isinstance(source, str) and source.strip():
                # Extract domain name for display text
                display_text = extract_domain_name(source)
                # Format as Telegram-compatible markdown link
                link_text = f"[{display_text}]({source})"
                message_parts.append(f"{i}. {link_text}")
    
    # Join all parts
    final_message = "\n".join(message_parts)
    
    # Ensure images is a list
    images_list = images if isinstance(images, list) else []
    
    return final_message, images_list


def escape_telegram_markdown(text: str) -> str:
    """
    Escape special characters for Telegram MarkdownV2 compatibility.
    
    Args:
        text (str): Original text to escape
        
    Returns:
        str: Escaped text compatible with Telegram
    """
    
    # First, convert headers (###, ##, #) to bold formatting
    text = convert_headers_to_bold(text)
    
    # Then, preserve existing markdown formatting by temporarily replacing it
    text = preserve_existing_markdown(text)
    
    # Escape special characters that could break Telegram parsing
    special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    
    # Note: Character escaping is commented out as it's not needed for current use case
    # for char in special_chars:
    #     text = text.replace(char, f'\\{char}')
    
    # Restore preserved markdown
    text = restore_preserved_markdown(text)
    
    return text


def convert_headers_to_bold(text: str) -> str:
    """Convert markdown headers (###, ##, #) to bold formatting.
    
    Args:
        text (str): Text with potential headers
    Returns:
        str: Text with headers converted to bold
    """
    import re
    
    # Store replacements to restore later
    replacements = {
        '**': '<<<BOLD>>>',
        '*': '<<<ITALIC>>>',
        '__': '<<<UNDERLINE>>>',
        '~~': '<<<STRIKE>>>',
        '`': '<<<CODE>>>',
    }
    
    # Temporarily replace existing markdown with placeholders
    for original, placeholder in replacements.items():
        text = text.replace(original, placeholder)
    
    # Pattern to match headers: 1-6 # symbols followed by space and text
    # Captures the header text without the # symbols
    header_pattern = r'^(#{1,6})\s+(.+)$'
    
    # Process each line
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        # Check if line is a header
        match = re.match(header_pattern, line.strip())
        if match:
            header_text = match.group(2).strip()
            # Convert to bold formatting
            processed_lines.append(f"**{header_text}**")
        else:
            processed_lines.append(line)
    
    # Rejoin lines
    result = '\n'.join(processed_lines)
    
    # Restore original markdown formatting
    for original, placeholder in replacements.items():
        result = result.replace(placeholder, original)
    
    return result


def preserve_existing_markdown(text: str) -> str:
    """
    Temporarily replace existing markdown with placeholders.
    """
    
    # Store replacements to restore later
    replacements = {
        '**': '<<<BOLD>>>',
        '*': '<<<ITALIC>>>',
        '__': '<<<UNDERLINE>>>',
        '~~': '<<<STRIKE>>>',
        '`': '<<<CODE>>>',
    }
    
    for original, placeholder in replacements.items():
        text = text.replace(original, placeholder)
    
    return text


def restore_preserved_markdown(text: str) -> str:
    """
    Restore preserved markdown formatting.
    """
    
    # Restore replacements
    replacements = {
        '<<<BOLD>>>': '*',
        '<<<ITALIC>>>': '_',
        '<<<UNDERLINE>>>': '__',
        '<<<STRIKE>>>': '~',
        '<<<CODE>>>': '`',
    }
    
    for placeholder, original in replacements.items():
        text = text.replace(placeholder, original)
    
    return text


def extract_domain_name(url: str) -> str:
    """
    Extract a clean domain name from URL for display.
    
    Args:
        url (str): Full URL
        
    Returns:
        str: Clean domain name
    """
    
    try:
        # Remove protocol
        if '://' in url:
            url = url.split('://', 1)[1]
        
        # Remove www prefix
        if url.startswith('www.'):
            url = url[4:]
        
        # Extract domain (before first slash)
        domain = url.split('/')[0]
        
        # Remove port if present
        domain = domain.split(':')[0]
        
        return domain
    
    except Exception:
        # Fallback to original URL if parsing fails
        return url[:50] + '...' if len(url) > 50 else url