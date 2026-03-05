'use strict';

import { Block, Div } from '@src/classes';

export class ImageDiv extends Div {

    constructor(attributes: any, public url: string, parent: Block) {

        super(attributes, parent);
    
        this.addClass('image');

        (async () => {

            const blob = await (await fetch(this.url)).blob();
            const url = URL.createObjectURL(blob);

            this.setStyle('background-image', `url(${url})`);

            setTimeout(() => {
                this.setData('displayed', 1);
            }, 50);

        })();
    }
}