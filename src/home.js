import { useNavigate } from "react-router-dom"


function Home() {
    const navigate = useNavigate();

    return (
        <div>
            Home with two buttons
            <button onClick={()=>
            {
                navigate("/details")
            
            }
            }>Details</button>

            
            <button onClick={()=>
            {
                navigate("/upload")
            }}>Upload</button>
        </div>
    )
}

export default Home
