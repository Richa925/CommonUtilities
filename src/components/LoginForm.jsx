import React, { useState } from 'react';
import Header from './Header';
import AuthMessage from './AuthMessage';
import './LoginForm.css';

function LoginForm() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [authenticated, setAuthenticated] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    // For now, authenticate with any values
    setAuthenticated(true);
    console.log('Login attempted with:', { username, password });
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleSubmit(e);
    }
  };

  return (
    <div className="login-container">
      <Header />
      <form className="login-form" onSubmit={handleSubmit}>
        <div className="form-group">
          <label htmlFor="username">Username</label>
          <input
            type="text"
            id="username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={handleKeyDown}
          />
        </div>
        <div className="form-group">
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={handleKeyDown}
          />
        </div>
        <div className="button-group">
          <button type="submit">Login</button>
          <button type="button" onClick={() => {
            setUsername('');
            setPassword('');
            setAuthenticated(false);
          }}>
            Reset
          </button>
        </div>
        <AuthMessage isVisible={authenticated} />
      </form>
    </div>
  );
}

export default LoginForm;