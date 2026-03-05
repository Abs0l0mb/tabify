'use strict';

export type TListenerTriggerCallback = (data: any) => void;
export type TListenerOffCallback = () => void;

export class Listener {

    constructor(
        private id: string,
        private event: string,
        private triggerCallback: TListenerTriggerCallback, 
        private offCallback: TListenerOffCallback
    ) {}

    /*
    **
    **
    */
    public getId() {

        return this.id;
    }

    /*
    **
    **
    */
    public getEvent() {

        return this.event;
    }

    /*
    **
    **
    */
    public trigger(data: any) {

        this.triggerCallback(data);
    }

    /*
    **
    **
    */
    public off() : void {

        this.offCallback();
    }
}