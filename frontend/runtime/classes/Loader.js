'use strict';

import * as readline from 'readline';

export class Loader {

	static loaders = {
		dot: {
			interval: 60,
			frames: [
				'⠁',
				'⠂',
				'⠄',
				'⡀',
				'⢀',
				'⠠',
				'⠐',
				'⠈'
			]
		},
		dots: {
			interval: 80,
			frames: [
				"⠋",
				"⠙",
				"⠹",
				"⠸",
				"⠼",
				"⠴",
				"⠦",
				"⠧",
				"⠇",
				"⠏"
			]
		},
		dots2: {
			interval: 80,
			frames: [
				"⣾",
				"⣽",
				"⣻",
				"⢿",
				"⡿",
				"⣟",
				"⣯",
				"⣷"
			]
		},
		line: {
			"interval": 90,
			"frames": [
				"-",
				"\\",
				"|",
				"/"
			]
		},
		arc: {
			"interval": 100,
			"frames": [
				"◜",
				"◠",
				"◝",
				"◞",
				"◡",
				"◟"
			]
		}
	}

	constructor(loaderKey = 'line') {

		this.started = false;
		this.intervalId = null;

		this.loaderKey = loaderKey;
		this.loader = Loader.loaders[this.loaderKey];
        this.loaderFrames = this.loader.frames;
        this.loaderTimeInterval = this.loader.interval;
	}

    /*
    **
    **
    */
    start() {

		if (this.started)
			return;

		this.started = true;

		process.stdout.write("\x1B[?25l");
		
        let index = 0

        this.intervalId = setInterval(() => {

			if (!this.started)
				return;

            let now = this.loaderFrames[index];

            if (now == undefined) {
                index = 0;
                now = this.loaderFrames[index];
            }

            process.stdout.write(now);
            readline.cursorTo(process.stdout, 0);

            index = index >= this.loaderFrames.length ? 0 : index + 1

        }, this.loaderTimeInterval);
    }

    /*
    **
    **
    */
    stop() {
		
		process.stdout.write('\x1B[?25h');

        this.started = false;

		clearInterval(this.intervalId);
	}
}