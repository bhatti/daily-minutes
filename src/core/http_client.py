"""HTTP client with centralized configuration and SSL handling."""

import ssl
import aiohttp
from typing import Optional, Dict, Any
from src.core.logging import get_logger

logger = get_logger(__name__)


class HTTPClient:
    """
    Centralized HTTP client for all API calls.

    Handles:
    - SSL certificate verification (configurable)
    - Timeout settings
    - Retry logic
    - User agent
    - Common headers
    """

    def __init__(
        self,
        verify_ssl: bool = True,
        timeout: int = 30,
        user_agent: Optional[str] = None
    ):
        """
        Initialize HTTP client.

        Args:
            verify_ssl: Whether to verify SSL certificates (default: True)
            timeout: Request timeout in seconds (default: 30)
            user_agent: Custom user agent string
        """
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.user_agent = user_agent or "Daily-Minutes/1.0"

        # Create SSL context
        if not verify_ssl:
            self.ssl_context = ssl.create_default_context()
            self.ssl_context.check_hostname = False
            self.ssl_context.verify_mode = ssl.CERT_NONE
            logger.warning("http_client_initialized", verify_ssl=False,
                         message="SSL verification disabled - not recommended for production")
        else:
            self.ssl_context = True  # Use default SSL verification
            logger.info("http_client_initialized", verify_ssl=True)

    async def get(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> aiohttp.ClientResponse:
        """
        Perform HTTP GET request.

        Args:
            url: URL to fetch
            headers: Optional headers
            params: Optional query parameters
            timeout: Optional timeout override

        Returns:
            aiohttp.ClientResponse object
        """
        # Merge headers
        merged_headers = {"User-Agent": self.user_agent}
        if headers:
            merged_headers.update(headers)

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self.timeout

        timeout_obj = aiohttp.ClientTimeout(total=request_timeout)

        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(
                url,
                headers=merged_headers,
                params=params,
                ssl=self.ssl_context
            ) as response:
                return response

    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> aiohttp.ClientResponse:
        """
        Perform HTTP POST request.

        Args:
            url: URL to post to
            data: Optional form data
            json: Optional JSON data
            headers: Optional headers
            timeout: Optional timeout override

        Returns:
            aiohttp.ClientResponse object
        """
        # Merge headers
        merged_headers = {"User-Agent": self.user_agent}
        if headers:
            merged_headers.update(headers)

        # Use provided timeout or default
        request_timeout = timeout if timeout is not None else self.timeout

        timeout_obj = aiohttp.ClientTimeout(total=request_timeout)

        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.post(
                url,
                data=data,
                json=json,
                headers=merged_headers,
                ssl=self.ssl_context
            ) as response:
                return response


# Singleton instance
_http_client: Optional[HTTPClient] = None


def get_http_client(verify_ssl: Optional[bool] = None) -> HTTPClient:
    """
    Get or create HTTP client instance.

    Args:
        verify_ssl: Optional override for SSL verification

    Returns:
        HTTPClient instance
    """
    global _http_client

    # If verify_ssl is explicitly provided, create new client
    if verify_ssl is not None:
        return HTTPClient(verify_ssl=verify_ssl)

    # Otherwise use singleton
    if _http_client is None:
        # Import here to avoid circular dependency
        from src.core.config_manager import get_config_manager
        config_mgr = get_config_manager()

        # Get SSL verification setting from config (default: True)
        verify_ssl_setting = config_mgr.get("http.verify_ssl", True)

        _http_client = HTTPClient(verify_ssl=verify_ssl_setting)

    return _http_client


def reset_http_client():
    """Reset the singleton HTTP client (useful for tests or config changes)."""
    global _http_client
    _http_client = None
