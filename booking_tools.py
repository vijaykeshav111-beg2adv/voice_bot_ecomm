"""LLM tool definitions and execution for the voice-enabled e-commerce assistant."""
import logging
import ecommerce
from knowledge_base import search_knowledge

BOOKING_ENABLED = True

def _fn(name, description, props, required):
    return {'type':'function','function':{'name':name,'description':description,'parameters':{'type':'object','properties':props,'required':required}}}
_STR={'type':'string'}
TOOLS=[
 _fn('search_products','Search the live product catalog by product name, category, or feature.',{'query':_STR,'category':_STR,'limit':{'type':'integer'}},['query']),
 _fn('get_product_details','Get current price, stock, SKU, and details for a product.',{'product':{'type':'string','description':'Product name or SKU'}},['product']),
 _fn('place_order','Place an order after reading back all products, quantities, total, customer name, and delivery address and receiving explicit confirmation.',{'customer_name':_STR,'customer_email':_STR,'address':_STR,'items':{'type':'array','items':{'type':'object','properties':{'sku':_STR,'name':_STR,'quantity':{'type':'integer'}},'required':['quantity']}},'confirmed':{'type':'boolean'}},['customer_name','address','items','confirmed']),
 _fn('track_order','Look up order status using the order reference.',{'order_ref':_STR},['order_ref']),
 _fn('search_store_knowledge','Search approved local store information for shipping, returns, payment, and product policies.',{'query':_STR,'top_k':{'type':'integer'}},['query']),
]
def system_prompt():
 return ("You are ShopMate, a voice shopping assistant for this e-commerce store. Your scope is shopping only: product discovery, live catalog details, cart/order assistance, order tracking, and store policies. "
 "If the customer asks an unrelated general-knowledge question (for example, 'What is Python?'), do not answer it; politely redirect them to shopping help in the customer's language. "
 "Understand Hindi, English, and Hinglish; reply in the language used by the customer. Keep spoken replies concise and natural. "
 "Use search_products for current catalog information and get_product_details for price or stock; never invent prices or availability. "
 "For shipping, returns, payment, privacy, or other store-policy questions, use search_store_knowledge and answer only from retrieved passages; if the documents do not contain the answer, say that the store documents do not specify it. "
 "For an order, confirm exact products/SKUs, quantities, total, customer name, and delivery address with the customer, then call place_order with confirmed=true only after a clear yes. "
 "Never claim an order exists unless the tool succeeds. Share the order reference and use track_order for status. "
 "Never ask for payment card numbers, CVV, banking passwords, or OTPs; this demo does not process payments. Do not claim real payment or shipping integration. "
 "Keep the current shopping task in context and do not discard pending order details because the customer asks an unrelated question.")

def run_tool(name,args,channel='voice'):
 try:
  if not isinstance(args,dict): return {'ok':False,'error':'bad_arguments','message':'Tool arguments must be an object.'}
  if name=='search_products':
   results=ecommerce.list_products(args.get('query',''),args.get('category',''),args.get('limit',10)); return {'ok':True,'products':results,'count':len(results),'say':f'Found {len(results)} matching products.'}
  if name=='get_product_details':
   p=ecommerce.get_product(args.get('product','')); return {'ok':bool(p),'product':p,'message':'Product found.' if p else 'Product not found.','say':(f"{p['name']} costs ₹{p['price']:.2f}; {p['stock']} in stock." if p else 'I could not find that product.')}
  if name=='place_order': return ecommerce.create_order(args.get('customer_name',''),args.get('customer_email',''),args.get('address',''),args.get('items',[]),args.get('confirmed') is True)
  if name=='track_order': return ecommerce.find_order(args.get('order_ref',''))
  if name=='search_store_knowledge': return search_knowledge(args.get('query',''),args.get('top_k',3))
  return {'ok':False,'error':'unknown_tool','message':f'Unknown tool: {name}'}
 except Exception:
  logging.exception('E-commerce tool failed: %s',name)
  return {'ok':False,'error':'system_unavailable','message':'The store service is temporarily unavailable. Please try again.'}
