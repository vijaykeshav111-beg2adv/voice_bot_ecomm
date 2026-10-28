import ecommerce

def test_catalog_search_returns_products():
    rows=ecommerce.list_products('headphones')
    assert rows and rows[0]['sku']=='SKU1001'
    assert rows[0]['price'] > 0

def test_order_requires_confirmation():
    result=ecommerce.create_order('Test Buyer','','123 Demo Road',[{'sku':'SKU1001','quantity':1}],False)
    assert result['ok'] is False
    assert result['error']=='confirmation_required'

def test_order_and_tracking():
    result=ecommerce.create_order('Test Buyer','','123 Demo Road',[{'sku':'SKU3001','quantity':1}],True)
    assert result['ok'] is True
    tracked=ecommerce.find_order(result['order']['order_ref'])
    assert tracked['ok'] is True
    assert tracked['order']['status']=='confirmed'
