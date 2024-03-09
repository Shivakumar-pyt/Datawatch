import React from 'react';
import './background.css';
import Background from './Background';
import Query from './Query';
import MyNavbar from './MyNavbar';
function HomePage() {
  return (
    <div className="container">
        <MyNavbar/>
      <Background/>
      <Query/>
    </div>
  );
}

export default HomePage;
