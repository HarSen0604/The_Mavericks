import ReactDOM from "react-dom/client";
import { BrowserRouter, Routes, Route } from "react-router-dom";


import Details from "./details";
import Upload from "./upload";
import Home from "./home";
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />}/>
        <Route index element={<Home />} />
        <Route path="upload" element={<Upload />} />
        <Route path="details" element={<Details />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
