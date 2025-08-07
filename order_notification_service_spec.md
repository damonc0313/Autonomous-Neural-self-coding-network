# Order Notification Service - Technical Specification

## 1. Service Responsibility

The **Order Notification Service** is responsible for:
- Consuming `order_created` events from the RabbitMQ message queue
- Sending confirmation emails to customers when orders are successfully created
- Maintaining delivery status and retry logic for failed notifications
- Providing health check endpoints for service monitoring

The service **does NOT**:
- Handle order processing or business logic
- Manage user authentication or authorization
- Store order details (references only)
- Send promotional or marketing emails
- Handle SMS or push notifications (email-only scope)

## 2. Core Logic Flow

### Event Processing Workflow

1. **Message Consumption**
   - Service connects to RabbitMQ and subscribes to the `order_created` queue
   - Validates incoming message structure against expected schema
   - Extracts order ID, customer email, and relevant order details

2. **User Data Enrichment**
   - Makes API call to User Service to fetch customer details (name, preferences)
   - Handles cases where user data is unavailable or incomplete

3. **Email Template Preparation**
   - Selects appropriate email template based on order type/customer segment
   - Populates template with order details and customer information
   - Generates personalized email content

4. **Email Delivery**
   - Sends email via SendGrid API
   - Handles delivery failures and logs appropriate metrics
   - Updates delivery status in local tracking

5. **Message Acknowledgment**
   - Acknowledges successful processing to RabbitMQ
   - Implements negative acknowledgment for failed processing

## 3. Data Models

### Pydantic Models

```python
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class OrderStatus(str, Enum):
    CREATED = "created"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"

class OrderItem(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    unit_price: float
    total_price: float

class OrderCreatedEvent(BaseModel):
    """Incoming message structure from RabbitMQ"""
    order_id: str = Field(..., description="Unique order identifier")
    customer_id: str = Field(..., description="Customer identifier")
    customer_email: EmailStr = Field(..., description="Customer email address")
    order_status: OrderStatus = Field(default=OrderStatus.CREATED)
    order_total: float = Field(..., ge=0, description="Total order amount")
    currency: str = Field(default="USD", description="Order currency")
    items: List[OrderItem] = Field(..., min_items=1, description="Order items")
    created_at: datetime = Field(..., description="Order creation timestamp")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class CustomerDetails(BaseModel):
    """Customer data from User Service"""
    customer_id: str
    first_name: str
    last_name: str
    email: EmailStr
    preferred_language: Optional[str] = "en"
    notification_preferences: Optional[Dict[str, bool]] = Field(default_factory=dict)

class NotificationStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"

class NotificationRecord(BaseModel):
    """Internal tracking model"""
    notification_id: str
    order_id: str
    customer_email: EmailStr
    status: NotificationStatus
    attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = "1.0.0"
    dependencies: Dict[str, str] = Field(default_factory=dict)
```

## 4. API Endpoints

### Health Check Endpoint

```python
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Order Notification Service", version="1.0.0")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint for service monitoring
    Returns service status and dependency health
    """
    dependencies = {}
    
    # Check RabbitMQ connection
    try:
        # Implement RabbitMQ health check
        dependencies["rabbitmq"] = "healthy"
    except Exception:
        dependencies["rabbitmq"] = "unhealthy"
    
    # Check SendGrid API
    try:
        # Implement SendGrid health check
        dependencies["sendgrid"] = "healthy"
    except Exception:
        dependencies["sendgrid"] = "unhealthy"
    
    # Check User Service connectivity
    try:
        # Implement User Service health check
        dependencies["user_service"] = "healthy"
    except Exception:
        dependencies["user_service"] = "unhealthy"
    
    overall_status = "healthy" if all(
        status == "healthy" for status in dependencies.values()
    ) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        dependencies=dependencies
    )

@app.get("/metrics")
async def metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    # Return metrics in Prometheus format
    pass
```

## 5. Key Dependencies

### External Libraries

```python
# requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic[email]==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
pika==1.3.2              # RabbitMQ client
sendgrid==6.10.0         # Email delivery
jinja2==3.1.2           # Email templating
aioredis==2.0.1         # Caching (optional)
structlog==23.2.0       # Structured logging
prometheus-client==0.19.0 # Metrics
python-multipart==0.0.6
httpx==0.25.2           # HTTP client for User Service
tenacity==8.2.3         # Retry logic
```

### Internal Service Interactions

1. **User Service API**
   - `GET /api/v1/users/{customer_id}` - Fetch customer details
   - Authentication: Internal service token
   - Timeout: 5 seconds with 3 retries

2. **RabbitMQ Message Queue**
   - Queue: `order_created`
   - Exchange: `orders`
   - Routing Key: `order.created`
   - Dead Letter Queue: `order_created_dlq`

### Configuration Management

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # RabbitMQ Configuration
    rabbitmq_url: str = "amqp://localhost:5672"
    rabbitmq_queue: str = "order_created"
    rabbitmq_dlq: str = "order_created_dlq"
    
    # SendGrid Configuration
    sendgrid_api_key: str
    sendgrid_from_email: str = "noreply@company.com"
    
    # User Service Configuration
    user_service_base_url: str = "http://user-service:8000"
    user_service_timeout: int = 5
    
    # Retry Configuration
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 60
    
    # Database Configuration (for notification tracking)
    database_url: str = "postgresql://localhost:5432/notifications"
    
    class Config:
        env_file = ".env"
```

## 6. Error Handling Strategy

### Retry Logic Implementation

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

class NotificationError(Exception):
    """Base exception for notification failures"""
    pass

class TemporaryNotificationError(NotificationError):
    """Temporary failures that should be retried"""
    pass

class PermanentNotificationError(NotificationError):
    """Permanent failures that should not be retried"""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(TemporaryNotificationError)
)
async def send_notification_with_retry(order_event: OrderCreatedEvent):
    """Send notification with exponential backoff retry"""
    try:
        await send_email_notification(order_event)
    except SendGridAPIError as e:
        if e.status_code >= 500:
            raise TemporaryNotificationError(f"SendGrid server error: {e}")
        else:
            raise PermanentNotificationError(f"SendGrid client error: {e}")
    except UserServiceError as e:
        if e.status_code >= 500:
            raise TemporaryNotificationError(f"User service unavailable: {e}")
        else:
            raise PermanentNotificationError(f"User not found: {e}")
```

### Dead Letter Queue Strategy

1. **Primary Queue Processing**
   - Messages are consumed from `order_created` queue
   - Failed messages are negatively acknowledged after max retries
   - Messages automatically move to `order_created_dlq`

2. **Dead Letter Queue Handling**
   - Separate consumer monitors DLQ for manual intervention
   - Failed messages are logged with full context
   - Alerts are sent to operations team for investigation

3. **Message TTL and Expiration**
   - Messages expire after 24 hours in primary queue
   - DLQ messages expire after 7 days
   - Expired messages are permanently discarded

### Error Categories and Responses

| Error Type | Retry Strategy | Action |
|------------|----------------|---------|
| Invalid message format | No retry | Log error, acknowledge message |
| User not found | No retry | Log warning, acknowledge message |
| SendGrid rate limit | Exponential backoff | Retry up to 3 times |
| SendGrid server error | Exponential backoff | Retry up to 3 times |
| Network timeout | Linear backoff | Retry up to 3 times |
| Database connection error | Exponential backoff | Retry up to 3 times |

## 7. Security & PII Concerns

### PII Data Handling

**Identified PII Elements:**
- Customer email addresses
- Customer first and last names
- Order details and purchase history

### Protection Measures

1. **Data Minimization**
   ```python
   # Only log order IDs and customer IDs, never email addresses
   logger.info("Processing order notification", 
               order_id=order_event.order_id,
               customer_id=order_event.customer_id)
   # NEVER: logger.info(f"Sending to {customer_email}")
   ```

2. **Secure Data Transmission**
   - All API calls to User Service use HTTPS with certificate validation
   - RabbitMQ connections use TLS encryption
   - SendGrid API calls use HTTPS with API key authentication

3. **Data Retention Policies**
   - Notification records retain only order_id and customer_id
   - Email addresses are not persisted beyond message processing
   - Logs are rotated every 7 days with PII scrubbing

4. **Access Controls**
   ```python
   # Environment-based configuration for sensitive data
   SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")  # Never hardcoded
   DATABASE_URL = os.getenv("DATABASE_URL")          # Connection strings secured
   ```

5. **Audit Logging**
   ```python
   # Structured logging without PII
   audit_logger.info("notification_sent", 
                     order_id=order_id,
                     customer_id=customer_id,
                     notification_id=notification_id,
                     timestamp=datetime.utcnow().isoformat())
   ```

6. **Error Message Sanitization**
   ```python
   def sanitize_error_message(error_msg: str, customer_email: str) -> str:
       """Remove PII from error messages before logging"""
       return error_msg.replace(customer_email, "[REDACTED_EMAIL]")
   ```

### Compliance Considerations

- **GDPR Compliance**: Customer email addresses are processed lawfully under legitimate interest for order fulfillment
- **Data Subject Rights**: Implement mechanisms to handle deletion requests
- **Breach Notification**: Automated alerts for failed email deliveries that might indicate security issues
- **Regular Security Audits**: Monthly reviews of logs and access patterns

### Security Headers and Configuration

```python
# Security middleware configuration
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "notification-service.internal"]
)

# Disable debug mode in production
app = FastAPI(debug=False, docs_url=None, redoc_url=None)
```

---

## Implementation Timeline

| Phase | Duration | Deliverables |
|-------|----------|-------------|
| Phase 1 | 1 week | Core message processing and basic email sending |
| Phase 2 | 3 days | Error handling and retry logic implementation |
| Phase 3 | 2 days | Health checks, metrics, and monitoring |
| Phase 4 | 2 days | Security hardening and PII compliance |
| Phase 5 | 1 day | Testing and deployment preparation |

**Total Estimated Duration: 2 weeks**