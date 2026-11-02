package auth

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"
	"unicode"
)

type APIKey struct {
	ID       string `json:"id"`
	TenantID string `json:"tenant_id"`
	Secret   string `json:"key"`
}

type identity struct {
	id       string
	tenantID string
	hash     [32]byte
}

type KeySet struct {
	byHash     map[[32]byte]identity
	single     identity
	singleOnly bool
}

func LoadAPIKeys(path string) (*KeySet, error) {
	entries, err := ReadAPIKeys(path)
	if err != nil {
		return nil, err
	}
	return NewKeySet(entries)
}

func ReadAPIKeys(path string) ([]APIKey, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read API key file: %w", err)
	}
	var entries []APIKey
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("parse API key file: %w", err)
	}
	for _, entry := range entries {
		if entry.ID == "" || entry.TenantID == "" || len(entry.Secret) < 16 {
			return nil, fmt.Errorf("each API key entry requires id, tenant_id, and a key of at least 16 characters")
		}
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("API key file must contain at least one key")
	}
	return entries, nil
}

func NewKeySet(entries []APIKey) (*KeySet, error) {
	if len(entries) == 0 {
		return nil, fmt.Errorf("at least one API key is required")
	}
	if len(entries) == 1 {
		key, err := makeIdentity(entries[0])
		if err != nil {
			return nil, err
		}
		return &KeySet{single: key, singleOnly: true}, nil
	}
	set := &KeySet{byHash: make(map[[32]byte]identity, len(entries))}
	for _, entry := range entries {
		key, err := makeIdentity(entry)
		if err != nil {
			return nil, err
		}
		if _, exists := set.byHash[key.hash]; exists {
			return nil, fmt.Errorf("duplicate API key secret")
		}
		set.byHash[key.hash] = key
	}
	return set, nil
}

func makeIdentity(entry APIKey) (identity, error) {
	if entry.ID == "" || entry.TenantID == "" || len(entry.Secret) < 16 {
		return identity{}, fmt.Errorf("each API key entry requires id, tenant_id, and a key of at least 16 characters")
	}
	return identity{id: entry.ID, tenantID: entry.TenantID, hash: sha256.Sum256([]byte(entry.Secret))}, nil
}

func (set *KeySet) Authenticate(header string) (tenantID, keyID string, ok bool) {
	token, valid := bearerToken(header)
	if !valid {
		return "", "", false
	}
	got := sha256.Sum256([]byte(token))
	if set.singleOnly {
		if subtle.ConstantTimeCompare(got[:], set.single.hash[:]) != 1 {
			return "", "", false
		}
		return set.single.tenantID, set.single.id, true
	}
	selected, found := set.byHash[got]
	if !found || subtle.ConstantTimeCompare(got[:], selected.hash[:]) != 1 {
		return "", "", false
	}
	return selected.tenantID, selected.id, true
}

func bearerToken(header string) (string, bool) {
	if scheme, token, separated := strings.Cut(header, " "); separated && strings.EqualFold(scheme, "Bearer") {
		token = strings.TrimSpace(token)
		if token == "" || hasUnicodeSpace(token) {
			return "", false
		}
		return token, true
	}
	fields := strings.Fields(header)
	if len(fields) != 2 || !strings.EqualFold(fields[0], "Bearer") {
		return "", false
	}
	return fields[1], true
}

func hasUnicodeSpace(value string) bool {
	for _, character := range value {
		if unicode.IsSpace(character) {
			return true
		}
	}
	return false
}

func ParseHMACKeys(value string) ([][]byte, error) {
	parts := strings.Split(value, ",")
	keys := make([][]byte, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			return nil, fmt.Errorf("session HMAC key list contains an empty value")
		}
		key, err := base64.RawURLEncoding.DecodeString(part)
		if err != nil {
			key, err = base64.URLEncoding.DecodeString(part)
		}
		if err != nil || len(key) < 32 {
			return nil, fmt.Errorf("session HMAC keys must be base64url values with at least 32 decoded bytes")
		}
		keys = append(keys, key)
	}
	return keys, nil
}

type sessionPayload struct {
	Version  string `json:"v"`
	TenantID string `json:"tenant"`
	Session  string `json:"session"`
	Model    string `json:"model,omitempty"`
	Expires  int64  `json:"exp"`
}

func SignSession(tenantID, upstreamID string, expires time.Time, keys [][]byte) (string, error) {
	return SignModelSession(tenantID, "", upstreamID, expires, keys)
}

func SignModelSession(tenantID, modelAlias, upstreamID string, expires time.Time, keys [][]byte) (string, error) {
	if tenantID == "" || upstreamID == "" || len(keys) == 0 || len(keys[0]) < 32 {
		return "", fmt.Errorf("session handle signing inputs are invalid")
	}
	payload, err := json.Marshal(sessionPayload{Version: "v1", TenantID: tenantID, Session: upstreamID, Model: modelAlias, Expires: expires.Unix()})
	if err != nil {
		return "", err
	}
	encoded := base64.RawURLEncoding.EncodeToString(payload)
	mac := hmac.New(sha256.New, keys[0])
	_, _ = mac.Write([]byte(encoded))
	signature := base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
	return encoded + "." + signature, nil
}

func VerifySession(handle, tenantID string, keys [][]byte, now time.Time) (string, bool) {
	session, valid := VerifyModelSession(handle, tenantID, keys, now)
	return session.UpstreamID, valid
}

type Session struct {
	UpstreamID string
	ModelAlias string
}

func VerifyModelSession(handle, tenantID string, keys [][]byte, now time.Time) (Session, bool) {
	parts := strings.Split(handle, ".")
	if len(parts) != 2 || len(keys) == 0 {
		return Session{}, false
	}
	signature, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return Session{}, false
	}
	valid := 0
	for _, key := range keys {
		mac := hmac.New(sha256.New, key)
		_, _ = mac.Write([]byte(parts[0]))
		valid |= boolInt(hmac.Equal(signature, mac.Sum(nil)))
	}
	if valid != 1 {
		return Session{}, false
	}
	payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[0])
	if err != nil {
		return Session{}, false
	}
	var payload sessionPayload
	if json.Unmarshal(payloadBytes, &payload) != nil || payload.Version != "v1" || payload.TenantID != tenantID || payload.Session == "" || payload.Expires <= now.Unix() {
		return Session{}, false
	}
	return Session{UpstreamID: payload.Session, ModelAlias: payload.Model}, true
}

func boolInt(value bool) int {
	if value {
		return 1
	}
	return 0
}
