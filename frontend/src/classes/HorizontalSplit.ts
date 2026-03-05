'use strict';

import {
    Block,
    Div,
    ClientLocation
} from '@src/classes';

export class HorizontalSplit extends Div {

    private leftZone: Div;
    private rightZone: Div;
    public leftContainer: Div;
    public rightContainer: Div;
    private resizer: Div;
    private x: number | null = null;
    
    constructor(parent?: Block) {

        super('split horizontal', parent);

        this.leftZone = new Div('left-zone', this);
        this.leftContainer = new Div('left-container', this.leftZone);
        this.resizer = new Div('resizer', this.leftZone);

        this.rightZone = new Div('right-zone', this);
        this.rightContainer = new Div('right-container', this.rightZone);

        this.resizer.onNative('mousedown', this.onDown.bind(this));
        document.addEventListener('mousemove', this.onMove.bind(this));
        document.addEventListener('mouseup', this.onUp.bind(this));

        this.resizer.onNative('touchstart', this.onDown.bind(this));
        document.addEventListener('touchmove', this.onMove.bind(this));
        document.addEventListener('touchend', this.onUp.bind(this));
        document.addEventListener('touchcancel', this.onUp.bind(this));

        this.leftContainer.on('append', this.onLeftAppend.bind(this));
        this.rightContainer.on('append', this.onRightAppend.bind(this));
    }

    /*
    **
    **
    */
    private onDown(event: MouseEvent | TouchEvent) : void {

        if (event instanceof MouseEvent)
            this.x = event.clientX;
        else
            this.x = event.touches[0].clientX;

        this.resizer.setData('resizing', 1);
        ClientLocation.get().block.setData('resizing-x', 1);
    }

    /*
    **
    **
    */
    private onMove(event: MouseEvent | TouchEvent) : void {

        if (!this.x)
            return;

        console.log(event);

        let x: number;

        if (event instanceof MouseEvent)
            x = event.clientX;
        else
            x = event.touches[0].clientX;

        let diff = this.x - x;

        if (diff === 0)
            return;
        
        let absDiff = Math.abs(diff);
        
        let parentWidth = this.element.parentElement.offsetWidth;
        let leftWidth = this.leftZone.element.offsetWidth;
        let rightWidth = this.rightZone.element.offsetWidth;

        let leftPercent: number;
        let rightPercent: number;

        if (diff > 0) {

            leftPercent = (leftWidth - absDiff) * 100 / parentWidth;
            rightPercent = (rightWidth + absDiff) * 100 / parentWidth;

            if (leftPercent <= 5) {
                leftPercent = 5;
                rightPercent = 95;
            }
        }
        else {

            leftPercent = (leftWidth + absDiff) * 100 / parentWidth;
            rightPercent = (rightWidth - absDiff) * 100 / parentWidth;

            if (rightPercent <= 5) {
                leftPercent = 95;
                rightPercent = 5;
            }
        }

        this.leftZone.setStyle('width', `${leftPercent}%`);
        this.rightZone.setStyle('width', `${rightPercent}%`);

        this.x = x;
    }

    /*
    **
    **
    */
    private onUp(event: MouseEvent | TouchEvent) : void {

        this.x = null;

        this.resizer.setData('resizing', 0);
        ClientLocation.get().block.setData('resizing-x', 0);
    }
    
    /*
    **
    **
    */
    public setLeftWidth(percent: number) {

        this.leftZone.setStyle('width', `${percent}%`);
        this.rightZone.setStyle('width', `${100 - percent}%`);
    }

    /*
    **
    **
    */
    public setRightWidth(percent: number) {
        
        this.leftZone.setStyle('width', `${100 - percent}%`);
        this.rightZone.setStyle('width', `${percent}%`);
    }

    /*
    **
    **
    */
    private onLeftAppend() : void {

        if (this.rightContainer.isEmpty()) {
            this.setLeftWidth(100);
            this.setRightWidth(0);
            this.setData('full-left', 1);
        }
    }

    /*
    **
    **
    */
    private onRightAppend() : void {

        if (parseInt(this.rightContainer.getStyle('width')) < 50) {
            this.setLeftWidth(50);
            this.setRightWidth(50);
        }

        this.setData('full-left', 0);
    }
}