// import Container from 'react-bootstrap/Container';
// import Navbar from 'react-bootstrap/Navbar';
// import 'bootstrap/dist/css/bootstrap.min.css'; 
// import './mynavbar.css';

// function MyNavbar() {
//   return (
//     <Navbar className="bg-body-tertiary topper ">
//       <Container>
//         <Navbar.Brand href="#home">Welcome</Navbar.Brand>
//         <Navbar.Toggle />
//         <Navbar.Collapse className="justify-content-end">
//           <Navbar.Text>
//             Signed in as: <a href="#login">Mark Otto</a>
//           </Navbar.Text>
//         </Navbar.Collapse>
//       </Container>
//     </Navbar>
//   );
// }

// export default MyNavbar;

import React from 'react';
import Container from 'react-bootstrap/Container';
import Navbar from 'react-bootstrap/Navbar';
import 'bootstrap/dist/css/bootstrap.min.css';
import './mynavbar.css';

// const isAdminUser = () => {
// //   backend
//   return true; 
// };

function MyNavbar() {
  const role_name = localStorage.getItem("role_name")
  return (
    <Navbar className="bg-body-info topper">
      <Container>
        <Navbar.Brand href="#home" style={{ color: 'white' }}>Welcome</Navbar.Brand>
        <Navbar.Toggle />
        <Navbar.Collapse className="justify-content-end">
          <Navbar.Text className="mb-0 h5" style={{ color: 'white' }}>
            {/* Signed in as: <a href="#login" style={{ color: 'white' }}>Mark Otto</a> */}
          </Navbar.Text>
          {role_name === "Administrator" && (
            <Navbar.Text>
                <div className="adminpart" style={{ color: 'white' }}>
                    <button>
              <a href="/warnings" >Admin Dashboard</a>
              </button>
              </div>
            </Navbar.Text>
          )}
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
}

export default MyNavbar;
