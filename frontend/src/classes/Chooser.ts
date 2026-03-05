'use strict';

import {
    Block,
    Div
} from '@src/classes';

export interface ChooserData {
    text: string,
    value: string
}

export class Chooser extends Div {

    public buttons: Div[] = [];
    public value: string;

    constructor(public chooserData: ChooserData[], parent: Block) {

        super('chooser', parent);

        for (let data of chooserData) {

            let button = new Div({
                'data-state': 0
            }, this)
                .write(data.text)
                .onNative('click', function() {

                    this.onButtonClicked(button.uid);
                    setTimeout(() => {
                        this.value = data.value;
                        this.emit(this.value);
                        this.emit('value', this.value);
                    }, 250);

                }.bind(this));

            button.publicData.chooserData = data;

            this.buttons.push(button);
        }

        this.onButtonClicked(this.buttons[0].uid);

        this.value = this.buttons[0].publicData.chooserData.value;
    }

    /*
    **
    **
    */
    private onButtonClicked(uid: string) : void {
        
        for (let button of this.buttons)
            button.setData('state', button.uid === uid ? 1 : 0);
    }
}