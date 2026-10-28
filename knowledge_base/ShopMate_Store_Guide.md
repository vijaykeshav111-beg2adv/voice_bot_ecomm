# ShopMate Store Guide

## Store overview
ShopMate is a demo online store. The live product catalog is the source of truth for current product names, SKUs, prices, descriptions, categories, and stock. The guide does not contain live inventory or prices.

## Ordering and confirmation
An order is created only after the customer reviews the exact products, quantities, total, customer name, and complete delivery address and explicitly confirms. A successful order returns an order reference beginning with ORD-. Keep the order reference for tracking.

## Payments
This demo does not process real payments and does not collect payment-card details. Payment methods and a payment gateway are not configured in this demo.

## Shipping and delivery
The demo stores the delivery address on the order but is not connected to a courier. It does not provide live shipment tracking or verified delivery estimates. Order tracking reports the status stored by this application.

## Sample returns and refunds policy
This is a sample policy for demonstration and testing only.

- **Return window:** Customers may request a return within 7 calendar days after the item is marked delivered.
- **Eligibility:** The item should be unused, undamaged, and returned with its original packaging, accessories, and proof of purchase. Items damaged through misuse are not eligible under this sample policy.
- **Wrong or damaged item:** Customers should report an incorrect or visibly damaged item within 48 hours of delivery and provide the order reference plus clear photos. The store team would review the request before approving a return or replacement.
- **Non-returnable items:** For this demo policy, personalized items, digital goods, and hygiene-sensitive items that have been opened are excluded unless the item arrived damaged or incorrect.
- **How to request a return:** Contact the demo support address help@shopmate.demo with the order reference, reason for return, and photos when relevant. Do not send payment-card details.
- **Refund timing:** After an approved return is received and inspected, a refund would typically be initiated within 5–7 business days. The time for funds to appear would depend on the original payment provider. This demo does not process real payments or refunds.
- **Cancellation:** A customer may request cancellation before an order is marked dispatched. Cancellation is not guaranteed after dispatch in this demo.
- **Return shipping:** The demo does not have a real courier or return-shipping service configured. Support would need to confirm the return method before anything is shipped.

These rules are illustrative. They must be reviewed and replaced by the merchant's actual policy before any production use.

## Privacy and security
The store should collect only the customer information needed to create an order. Customer order information should not be revealed to other customers. Public production deployment requires authentication and HTTPS.

## Assistant scope
ShopMate helps customers discover products, check live catalog details, place confirmed demo orders, track order references, and answer questions using approved store documents. When a policy detail is not present in the indexed documents, the assistant should say it cannot verify the detail instead of inventing an answer.
