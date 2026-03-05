'use strict';

import {
    ContextBox,
    Div
} from '@src/classes';

export interface ContextMenuOption {
    text: string,
    callback: () => void
}

export class ContextMenu extends ContextBox {

    constructor(protected x: number, protected y: number, protected options: ContextMenuOption[]) {
        
        super(x, y);

        this.addClass('context-menu');
        
        for (let option of this.options) {

            new Div('item', this.box).write(option.text).onNative('click', () => {
                option.callback();
                this.hide();
            });
        }

        this.show();
    }
}