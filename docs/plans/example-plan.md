# AUTH-1: Firebase Admin SDK Implementation Plan (Example)


**Goal:** Replace Supabase service layer with Firebase Admin SDK initialization in auth-service.

**Architecture:** `FirebaseAuthService` wraps Firebase Admin SDK, exposing `Auth()` method for handlers to use SDK directly. Config simplified to Firebase vars only. Code will not compile until AUTH-3+ handler migrations are complete.

**Tech Stack:** Go 1.21, Firebase Admin SDK (`firebase.google.com/go/v4`), gorilla/mux

**Spec:** [AUTH-1 Design Spec](../specs/example-spec.md)

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `go.mod` | Add Firebase SDK dependencies |
| Modify | `config/config.go` | Remove Supabase vars, add `FirebaseProjectID` |
| Delete | `service/supabase.go` | Old Supabase service (replaced) |
| Create | `service/firebase.go` | Firebase initialization + `Auth()` accessor |
| Modify | `cmd/api/main.go` | Use `FirebaseAuthService`, handle init error |
| Modify | `handlers/auth.go` | Update type refs to `FirebaseAuthService` |
| Modify | `handlers/admin.go` | Update type refs to `FirebaseAuthService` |
| Modify | `handlers/oauth.go` | Update type refs to `FirebaseAuthService` |
| Modify | `handlers/user.go` | Update type refs to `FirebaseAuthService` |

---

## Task 1: Add Firebase SDK Dependencies

**Files:**
- Modify: `go.mod`

- [ ] **Step 1: Add Firebase imports to go.mod**

```bash
cd /Users/anshulchiranth/Desktop/Strike/intelupsell/.worktrees/auth1-firebase-sdk/auth-service
go get firebase.google.com/go/v4
go get google.golang.org/api/option
```

- [ ] **Step 2: Verify dependencies added**

Run: `cat go.mod | grep -E "(firebase|google.golang.org/api)"`

Expected: Lines showing `firebase.google.com/go/v4` and `google.golang.org/api`

---

## Task 2: Update Config

**Files:**
- Modify: `config/config.go`

- [ ] **Step 1: Replace config struct and Load function**

Replace entire contents of `config/config.go` with:

```go
// config/config.go
package config

import (
	"log"
	"os"
)

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

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}
```

---

## Task 3: Create Firebase Service

**Files:**
- Delete: `service/supabase.go`
- Create: `service/firebase.go`

- [ ] **Step 1: Delete supabase.go**

```bash
rm /Users/anshulchiranth/Desktop/Strike/intelupsell/.worktrees/auth1-firebase-sdk/auth-service/service/supabase.go
```

- [ ] **Step 2: Create firebase.go**

Create `service/firebase.go` with:

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

## Task 4: Update main.go

**Files:**
- Modify: `cmd/api/main.go`

- [ ] **Step 1: Update service initialization**

In `cmd/api/main.go`, replace:

```go
// Create Supabase service
supabaseService := service.NewSupabaseAuthService(cfg)

// Create handlers
authHandler := handlers.NewAuthHandler(supabaseService)
userHandler := handlers.NewUserHandler(supabaseService)
adminHandler := handlers.NewAdminHandler(supabaseService)
oauthHandler := handlers.NewOAuthHandler(supabaseService)
```

With:

```go
// Create Firebase service
firebaseService, err := service.NewFirebaseAuthService(cfg)
if err != nil {
	log.Fatalf("Failed to initialize Firebase service: %v", err)
}

// Create handlers
authHandler := handlers.NewAuthHandler(firebaseService)
userHandler := handlers.NewUserHandler(firebaseService)
adminHandler := handlers.NewAdminHandler(firebaseService)
oauthHandler := handlers.NewOAuthHandler(firebaseService)
```

- [ ] **Step 2: Update log message**

Replace:

```go
log.Printf("Supabase Auth Service starting on port %s", cfg.Port)
log.Printf("Supabase URL: %s", cfg.SupabaseURL)
```

With:

```go
log.Printf("Firebase Auth Service starting on port %s", cfg.Port)
log.Printf("Firebase Project: %s", cfg.FirebaseProjectID)
```

---

## Task 5: Update Handler Type References

**Files:**
- Modify: `handlers/auth.go`
- Modify: `handlers/admin.go`
- Modify: `handlers/oauth.go`
- Modify: `handlers/user.go`

- [ ] **Step 1: Update handlers/auth.go**

Replace:

```go
type AuthHandler struct {
	service *service.SupabaseAuthService
}

func NewAuthHandler(service *service.SupabaseAuthService) *AuthHandler {
	return &AuthHandler{service: service}
}
```

With:

```go
type AuthHandler struct {
	service *service.FirebaseAuthService
}

func NewAuthHandler(service *service.FirebaseAuthService) *AuthHandler {
	return &AuthHandler{service: service}
}
```

- [ ] **Step 2: Update handlers/admin.go**

Replace:

```go
type AdminHandler struct {
	service *service.SupabaseAuthService
}

func NewAdminHandler(service *service.SupabaseAuthService) *AdminHandler {
	return &AdminHandler{service: service}
}
```

With:

```go
type AdminHandler struct {
	service *service.FirebaseAuthService
}

func NewAdminHandler(service *service.FirebaseAuthService) *AdminHandler {
	return &AdminHandler{service: service}
}
```

- [ ] **Step 3: Update handlers/oauth.go**

Replace:

```go
type OAuthHandler struct {
	service *service.SupabaseAuthService
}

func NewOAuthHandler(service *service.SupabaseAuthService) *OAuthHandler {
	return &OAuthHandler{service: service}
}
```

With:

```go
type OAuthHandler struct {
	service *service.FirebaseAuthService
}

func NewOAuthHandler(service *service.FirebaseAuthService) *OAuthHandler {
	return &OAuthHandler{service: service}
}
```

- [ ] **Step 4: Update handlers/user.go**

Replace:

```go
type UserHandler struct {
	service *service.SupabaseAuthService
}

func NewUserHandler(service *service.SupabaseAuthService) *UserHandler {
	return &UserHandler{service: service}
}
```

With:

```go
type UserHandler struct {
	service *service.FirebaseAuthService
}

func NewUserHandler(service *service.FirebaseAuthService) *UserHandler {
	return &UserHandler{service: service}
}
```

---

## Task 6: Run go mod tidy

**Files:**
- Modify: `go.mod`, `go.sum`

- [ ] **Step 1: Run go mod tidy**

```bash
cd /Users/anshulchiranth/Desktop/Strike/intelupsell/.worktrees/auth1-firebase-sdk/auth-service
go mod tidy
```

- [ ] **Step 2: Verify go.mod looks correct**

Run: `cat go.mod`

Expected: Should include `firebase.google.com/go/v4` and related dependencies.

**Note:** Code will NOT compile at this point. Handler method bodies still reference removed Supabase methods (`MakeRequest`, etc.). This is expected — AUTH-3 through AUTH-8 will update the handler logic.

---

## Task 7: Commit Changes

- [ ] **Step 1: Stage all changes**

```bash
cd /Users/anshulchiranth/Desktop/Strike/intelupsell/.worktrees/auth1-firebase-sdk
git add -A
```

- [ ] **Step 2: Review staged changes**

```bash
git status
git diff --cached --stat
```

Expected files:
- `auth-service/go.mod` (modified)
- `auth-service/go.sum` (modified)
- `auth-service/config/config.go` (modified)
- `auth-service/service/supabase.go` (deleted)
- `auth-service/service/firebase.go` (new)
- `auth-service/cmd/api/main.go` (modified)
- `auth-service/handlers/auth.go` (modified)
- `auth-service/handlers/admin.go` (modified)
- `auth-service/handlers/oauth.go` (modified)
- `auth-service/handlers/user.go` (modified)
- `docs/superpowers/specs/2026-04-08-auth1-firebase-sdk-design.md` (new)
- `docs/superpowers/plans/2026-04-08-auth1-firebase-sdk.md` (new)


---

## Post-Implementation State

After completing these tasks:
- Firebase Admin SDK is added to dependencies
- `FirebaseAuthService` is initialized with `FIREBASE_PROJECT_ID` and `GOOGLE_APPLICATION_CREDENTIALS`
- All handler type references updated
- **Code does not compile** — handler method bodies still call removed Supabase methods
- Ready for AUTH-3 through AUTH-8 to migrate handler logic
