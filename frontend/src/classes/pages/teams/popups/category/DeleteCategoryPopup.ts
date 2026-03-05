'use strict';

import {
    Popup,
    Api
} from '@src/classes';

export class DeleteCategoryPopup extends Popup {

    constructor(private id: number) {

        super({
            validText: 'Delete',
            validRed: true,
            cancellable: true,
            title: `Delete category confirmation`,
            message: `Do you want to delete category ${id}?`
        });

        this.ready();
    }

    /*
    **
    **
    */
    public async onValid() : Promise<void> {

        this.validButton.load();

        try {

            await Api.post('/category/delete', {
                id: this.id
            });

            this.hide();
            
            this.emit('done');
            
        } catch(error: any) {

            console.log(error);
            
            this.unlock();
            this.validButton.unload();
        }
    }
}