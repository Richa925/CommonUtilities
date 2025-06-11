import React from 'react';

function AuthMessage({ isVisible }) {
  if (!isVisible) return null;
  
  return (
    <div className="auth-message success">
      Authentication successful!
    </div>
  );
}

export default AuthMessage;