import React, { useState } from 'react';
import { Image } from 'cloudinary-react';


const Upload = () => {
  const [image, setImage] = useState('');
  const [url,setUrl] = useState('');


  const uploadImage = async (e) => {
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append('file', file);
    formData.append('upload_preset', 'oskvfu3n'); // Replace 'your_upload_preset' with your Cloudinary upload preset name
    const res = await fetch(
      `https://api.cloudinary.com/v1_1/disd40wdk/image/upload`, // Using provided cloud name 'disd40wdk'
      {
        method: 'POST',
        body: formData,
      }
    );
    const data = await res.json();
    setImage(data.secure_url);
    setUrl(data.url)

    console.log(url+"this is the url")
    try {
        const response = await fetch('http://127.0.0.1:5000/uploadUrl', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ url })
        });
  
        if (response.ok) {
          console.log('Request sent successfully!');
        } else {
          console.error('Failed to send request.');
        }
      } catch (error) {
        console.error('Error sending request:', error);
      }

  };

  return (
    
    <div>
      <input type="file" onChange={uploadImage} />
      {image && (
        <div>
          <Image cloudName="disd40wdk" publicId={image} /> {/* Using provided cloud name 'disd40wdk' */}
        </div>
      )}
    </div>
  );
};

export default Upload;
