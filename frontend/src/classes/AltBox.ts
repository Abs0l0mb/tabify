'use strict';

import {
    Block,
    Div,
    ClientLocation,
    Tools
} from '@src/classes';

export interface AltBoxOptions {
    className?: string,
    fixed?: boolean
}

export class AltBox {

    private box: Div;

    constructor(private targetBlock: Block, private text: string, private options: AltBoxOptions = {}) {

        targetBlock!.onNative('mouseenter', this.onMouseEnter.bind(this));
        targetBlock!.onNative('mouseleave', this.onMouseLeave.bind(this));

        if (!this.options.fixed)
            targetBlock!.onNative('mousemove', this.onMouseMove.bind(this));
    }

    /*
    **
    **
    */
    private onMouseEnter(event: MouseEvent) : void {

        this.box = new Div(`alt-box ${this.options.className}`, ClientLocation.get().block).write(this.text);

        this.onMouseMove(event);
    }

    /*
    **
    **
    */
    private onMouseMove(event: MouseEvent) : void {

        const offsetX = 14;
        const offsetY = 4;

        let x: number = event.clientX + offsetX;
        let y: number = event.clientY + offsetY;

        if (x + this.box.element.offsetWidth >= document.body.scrollWidth) {
            x -= this.box.element.offsetWidth - offsetX;
            if (x < 0)
                x = offsetX;
        }

        if (y + this.box.element.offsetHeight >= document.body.scrollHeight) {
            y -= this.box.element.offsetHeight - offsetY;
            if (y < 0)
                y = offsetY;
        }

        this.setPosition(x, y);
    }

    /*
    **
    **
    */
    private onMouseLeave() : void {
        
        this.box!.delete();
    }

    /*
    **
    **
    */
    private setPosition(x: number, y: number) : void {

        this.box!.setStyles({
            left: `${x}px`,
            top: `${y}px`
        });
    }

    /*
    **
    **
    */
    public setText(text: string) : void {

        this.text = text;
    }
}