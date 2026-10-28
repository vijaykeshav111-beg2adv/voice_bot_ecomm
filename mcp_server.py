"""MCP server exposing the demo store catalog, orders, and local RAG knowledge."""
import os
from mcp.server.fastmcp import FastMCP
import booking_tools
import ecommerce
mcp=FastMCP(os.getenv('MCP_SERVER_NAME','voicebot-ecommerce'))
@mcp.tool()
def search_products(query: str, category: str = '', limit: int = 10) -> dict:
    """Search the current product catalog by name, category, SKU, or description."""
    return booking_tools.run_tool('search_products',{'query':query,'category':category,'limit':limit},channel='mcp')
@mcp.tool()
def get_product_details(product: str) -> dict:
    """Return the live demo catalog price and stock for a product name or SKU."""
    return booking_tools.run_tool('get_product_details',{'product':product},channel='mcp')
@mcp.tool()
def place_order(customer_name: str, address: str, items: list, confirmed: bool, customer_email: str = '') -> dict:
    """Place an order only after customer explicitly confirms the items, total, and address."""
    return booking_tools.run_tool('place_order',{'customer_name':customer_name,'customer_email':customer_email,'address':address,'items':items,'confirmed':confirmed},channel='mcp')
@mcp.tool()
def track_order(order_ref: str) -> dict:
    """Look up the current demo order status by order reference."""
    return booking_tools.run_tool('track_order',{'order_ref':order_ref},channel='mcp')
@mcp.tool()
def search_store_knowledge(query: str, top_k: int = 3) -> dict:
    """Retrieve approved local store policy passages using RAG."""
    return booking_tools.run_tool('search_store_knowledge',{'query':query,'top_k':top_k},channel='mcp')
@mcp.tool()
def list_recent_orders(limit: int = 20) -> dict:
    """List recent demo orders. Use only in a trusted/admin context; this does not enforce authentication."""
    return {'ok':True,'orders':ecommerce.list_orders(limit)}
if __name__=='__main__':
    ecommerce.init_db(); mcp.run()
