import { useEffect, useState } from 'react';
import './App.css';

const WS_URL = 'ws://127.0.0.1:5678/';
const urls = new WeakMap();

const blobUrl = (blob: any) => {
	if (urls.has(blob)) {
		return urls.get(blob);
	}
	const url = URL.createObjectURL(blob);
	urls.set(blob, url);
	return url;
}

function App(props: any) {
	const [imgURL, setImgURL] = useState<string>('');

	useEffect(() => {
		const ws = new WebSocket(WS_URL);
		ws.onmessage = (event) => {
			if (event.data) {
				setImgURL(blobUrl(event.data));
			}
		};
	}, []);

	return (
		<div className="App">
			<img id="camera" alt="Camera Live" src={imgURL} />
		</div>
	);
}

export default App;
