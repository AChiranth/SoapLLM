# AUTH-1: Add Firebase Admin SDK for Go (Example)

**Date:** 2026-04-08
**Status:** Draft
**Parent Plan:** List parent plan here

---

## Overview

Add the Firebase Admin SDK to the auth-service Go application, replacing the Supabase service layer. This is foundation work for AUTH-3 through AUTH-8, which will update the handler logic.

**Scope:** Service layer and config only. Handlers will not compile until subsequent tasks are completed.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Environment variables | Additive (keep Supabase vars in .env, add Firebase) | Smooth transition |
| Code structure | In-place replacement (no new files) | User preference |
| Service naming | `FirebaseAuthService` | Explicit about provider |
| File rename | `supabase.go` → `firebase.go` | Matches struct name, avoids confusion with `handlers/auth.go` |
| Firebase init | Required at startup | Service won't run until migration complete |
| Service abstraction | Expose `*auth.Client` directly via `Auth()` method | Handlers use SDK directly, less wrapper code |
| Compile state | Non-compiling until AUTH-3+ complete | Acceptable since no tests/users until migration done |

---

## File Changes

### 1. `service/supabase.go` → `service/firebase.go`

**Remove:**
- `SupabaseAuthService` struct
- `NewSupabaseAuthService()` constructor
- `MakeRequest()` method
- `MakeAuthenticatedRequest()` method
- `MakeFormRequest()` method
- `MakeRestRequest()` method

**Add:**

```go
package service

import (
    "context"
    "fmt"
    "net/http"
    "os"
    "time"

    firebase "firebase.google.com/go/v4"
    "firebase.google.com/go/v4/auth"
    "google.golang.org/api/option"

    "github.com/Strike-Bet/betting-engine/auth-service/config"
)

type FirebaseAuthService struct {
    config      *config.Config
    httpClient  *http.Client
    firebaseApp *firebase.App
    authClient  *auth.Client
}

func NewFirebaseAuthService(cfg *config.Config) (*FirebaseAuthService, error) {
    ctx := context.Background()
    
    opt := option.WithCredentialsFile(os.Getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    
    app, err := firebase.NewApp(ctx, &firebase.Config{
        ProjectID: cfg.FirebaseProjectID,
    }, opt)
    if err != nil {
        return nil, fmt.Errorf("failed to initialize firebase app: %w", err)
    }

    authClient, err := app.Auth(ctx)
    if err != nil {
        return nil, fmt.Errorf("failed to get firebase auth client: %w", err)
    }

    return &FirebaseAuthService{
        config:      cfg,
        httpClient:  &http.Client{Timeout: 30 * time.Second},
        firebaseApp: app,
        authClient:  authClient,
    }, nil
}

func (s *FirebaseAuthService) Auth() *auth.Client {
    return s.authClient
}

func (s *FirebaseAuthService) GetConfig() *config.Config {
    return s.config
}
```

---

### 2. `config/config.go`

**Remove:**
- `SupabaseURL` field
- `SupabaseKey` field
- `ServiceRoleKey` field
- `JWTSecret` field
- Validation for Supabase vars

**Update to:**

```go
type Config struct {
    FirebaseProjectID string
    Port              string
    Environment       string
}

func Load() *Config {
    config := &Config{
        FirebaseProjectID: getEnv("FIREBASE_PROJECT_ID", ""),
        Port:              getEnv("PORT", "8081"),
        Environment:       getEnv("ENVIRONMENT", "development"),
    }

    if config.FirebaseProjectID == "" {
        log.Fatal("FIREBASE_PROJECT_ID environment variable is required")
    }

    return config
}
```

---

### 3. `cmd/api/main.go`

**Update service initialization:**

```go
// Before
supabaseService := service.NewSupabaseAuthService(cfg)

// After
firebaseService, err := service.NewFirebaseAuthService(cfg)
if err != nil {
    log.Fatalf("Failed to initialize Firebase service: %v", err)
}
```

**Update handler creation to use `firebaseService`.**

---

### 4. `handlers/*.go`

**Update type references only (method bodies unchanged, will break compilation):**

```go
// Before
type AuthHandler struct {
    service *service.SupabaseAuthService
}

func NewAuthHandler(service *service.SupabaseAuthService) *AuthHandler

// After
type AuthHandler struct {
    service *service.FirebaseAuthService
}

func NewAuthHandler(service *service.FirebaseAuthService) *AuthHandler
```

Same pattern for `AdminHandler`, `OAuthHandler`, `UserHandler`.

---

### 5. `go.mod`

**Add dependencies:**

```
firebase.google.com/go/v4
google.golang.org/api
```

Run `go mod tidy` after adding imports.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FIREBASE_PROJECT_ID` | Yes | GCP project ID (e.g., `intelupsell-staging`) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes | Path to service account JSON file |
| `PORT` | No | Server port (default: 8081) |
| `ENVIRONMENT` | No | Environment name (default: development) |

---

## Acceptance Criteria

From parent plan:
- [x] Firebase Admin SDK added to dependencies
- [x] Firebase client initializes successfully with service account
- [x] Can verify Firebase ID tokens (via `Auth().VerifyIDToken()`)
- [x] Can create users via Firebase Admin SDK (via `Auth().CreateUser()`)

---

## Post-AUTH-1 State

After this task:
- Service layer fully migrated to Firebase
- Config uses Firebase vars only
- **Code does not compile** — handlers still reference removed Supabase methods
- Must complete AUTH-3 through AUTH-8 before service can run

---

## Testing

No tests until handlers are migrated. Manual verification:
1. `go mod tidy` succeeds
2. Service initializes with valid Firebase credentials (after handler migration)
