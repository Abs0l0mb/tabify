'use strict';

import {
    Block,
    Div,
    ClientLocation
} from '@src/classes';

export class VerticalSplit extends Div {

    public topContainer: Div;
    public bottomContainer: Div;
    private topZone: Div;
    private bottomZone: Div;
    private resizer: Div;
    private y: number | null = null;

    constructor(parent?: Block) {

        super('split vertical', parent);

        this.topZone = new Div('top-zone', this);
        this.topContainer = new Div('top-container', this.topZone);
        this.resizer = new Div('resizer', this.topZone);

        this.bottomZone = new Div('bottom-zone', this);
        this.bottomContainer = new Div('bottom-container', this.bottomZone);

        this.resizer.onNative('mousedown', this.onDown.bind(this));
        document.addEventListener('mousemove', this.onMove.bind(this));
        document.addEventListener('mouseup', this.onUp.bind(this));

        this.resizer.onNative('touchstart', this.onDown.bind(this));
        document.addEventListener('touchmove', this.onMove.bind(this));
        document.addEventListener('touchend', this.onUp.bind(this));
        document.addEventListener('touchcancel', this.onUp.bind(this));

        this.topContainer.on('append', this.onTopAppend.bind(this));
        this.bottomContainer.on('append', this.onBottomAppend.bind(this));
    }

    /*
    **
    **
    */
    private onDown(event: MouseEvent | TouchEvent) : void {

        if (event instanceof MouseEvent)
            this.y = event.clientY;
        else
            this.y = event.touches[0].clientY;
        
        this.resizer.setData('resizing', 1);
        ClientLocation.get().block.setData('resizing-y', 1);
    }

    /*
    **
    **
    */
    private onMove(event: MouseEvent | TouchEvent) : void {

        if (!this.y)
            return;

        let y: number;

        if (event instanceof MouseEvent)
            y = event.clientY;
        else
            y = event.touches[0].clientY;

        let diff = this.y - y;

        if (diff === 0)
            return;
        
        let absDiff = Math.abs(diff);

        let parentHeight = this.element.parentElement.offsetHeight;
        let topHeight = this.topZone.element.offsetHeight;
        let bottomHeight = this.bottomZone.element.offsetHeight;

        let topPercent: number;
        let bottomPercent: number;

        if (diff > 0) {

            topPercent = (topHeight - absDiff) * 100 / parentHeight;
            bottomPercent = (bottomHeight + absDiff) * 100 / parentHeight;

            if (topPercent <= 10) {
                topPercent = 10;
                bottomPercent = 90;
            }
        }
        else {

            topPercent = (topHeight + absDiff) * 100 / parentHeight;
            bottomPercent = (bottomHeight - absDiff) * 100 / parentHeight;
            
            if (bottomPercent <= 10){
                topPercent = 90;
                bottomPercent = 10;
            }
        }

        this.topZone.setStyle('height', `${topPercent}%`);
        this.bottomZone.setStyle('height', `${bottomPercent}%`);
        
        this.y = y;
    }

    /*
    **
    **
    */
    private onUp(event: MouseEvent | TouchEvent) : void {

        this.y = null;

        this.resizer.setData('resizing', 0);
        ClientLocation.get().block.setData('resizing-y', 0);
    }

    /*
    **
    **
    */
    public setTopHeight(percent: number) {

        this.topZone.setStyle('height', `${percent}%`);
        this.bottomZone.setStyle('height', `${100 - percent}%`);
    }

    /*
    **
    **
    */
    public setBottomHeight(percent: number) {
        
        this.topZone.setStyle('height', `${100 - percent}%`);
        this.bottomZone.setStyle('height', `${percent}%`);
    }

    /*
    **
    **
    */
    private onTopAppend() : void {

        if (this.bottomContainer.isEmpty()) {
            this.setTopHeight(100);
            this.setBottomHeight(0);
            this.setData('full-top', 1);
        }
    }

    /*
    **
    **
    */
    private onBottomAppend() : void {

        if (parseInt(this.bottomContainer.getStyle('height')) < 50) {
            this.setTopHeight(50);
            this.setBottomHeight(50);
        }

        this.setData('full-top', 0);
    }
}